#ifndef KET_MPI_GATE_GATE_HPP
# define KET_MPI_GATE_GATE_HPP

# include <tuple>
# include <array>
# include <vector>
# include <iterator>
# include <utility>

# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/buffer_range.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace gate_detail
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
          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          using state_integer_type = ::ket::meta::state_integer_t<Qubit>;
          using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;

#   ifndef KET_DEFAULT_NUM_ON_CACHE_QUBITS
#     define KET_DEFAULT_NUM_ON_CACHE_QUBITS 16
#   endif // KET_DEFAULT_NUM_ON_CACHE_QUBITS
          constexpr auto num_on_cache_qubits = bit_integer_type{KET_DEFAULT_NUM_ON_CACHE_QUBITS};
          constexpr auto on_cache_state_size = ::ket::utility::integer_exp2<state_integer_type>(num_on_cache_qubits);
          assert(on_cache_state_size <= data_block_size);

          // ggggg|uuuu|xxxx|yyyy|zzzzzz: (global+unit+local) qubits
          // xxxx|yyyy|zzzzzz: local qubits
          // * xxxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits

          // Case 1) All operated qubits are on-cache qubits
          if (::ket::utility::all_in_state_vector(num_on_cache_qubits, permutated_qubit.qubit(), permutated_qubits.qubit()...))
          {
            constexpr auto num_operated_qubits = sizeof...(Qubits) + 1u;
            std::array<state_integer_type, num_operated_qubits> qubit_masks{};
            ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
            std::array<state_integer_type, num_operated_qubits + 1u> index_masks{};
            ::ket::gate::gate_detail::make_index_masks(index_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);

            //TODO
            // Assume on-cache qubits are less significant than page qubits
            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, &qubit_masks, &index_masks, &function](auto const first, auto const last)
              {
                for (auto iter = first; iter < last; iter += on_cache_state_size)
                  ::ket::gate::gate_detail::gate(parallel_policy, iter, iter + on_cache_state_size, qubit_masks, index_masks, function);
              });
          }

          // Case 2) Some of the operated qubits are off-cache qubits
          //TODO: present impl. is not page-aware
          auto const present_buffer_size = static_cast<state_integer_type>(::ket::mpi::utility::buffer_end(local_state, buffer) - ::ket::mpi::utility::buffer_begin(local_state, buffer));

          if (present_buffer_size >= on_cache_state_size)
          {
            auto const first = begin(local_state);
            auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
            for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
              ::ket::gate::cache::unsafe::gate(
                parallel_policy,
                first + data_block_index * data_block_size,
                first + (data_block_index + 1u) * data_block_size,
                buffer_first, buffer_first + on_cache_state_size,
                function, permutated_qubit.qubit(), permutated_qubits.qubit()...);
          }
          else // if (present_buffer_size >= on_cache_state_size)
          {
            buffer.resize(on_cache_state_size);

            auto const first = begin(local_state);
            for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
              ::ket::gate::cache::unsafe::gate(
                parallel_policy,
                first + data_block_index * data_block_size,
                first + (data_block_index + 1u) * data_block_size,
                begin(buffer), end(buffer),
                function, permutated_qubit.qubit(), permutated_qubits.qubit()...);
          }

          return local_state;
        }
# endif // KET_USE_ON_CACHE_STATE_VECTOR
      } // namespace gate_detail

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

        return ::ket::mpi::gate::gate_detail::gate(
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

        return ::ket::mpi::gate::gate_detail::gate(
          mpi_policy, parallel_policy, local_state, communicator, environment,
          std::forward<Function>(function), permutation[std::forward<Qubit>(qubit)], permutation[std::forward<Qubits>(qubits)]...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_GATE_HPP
