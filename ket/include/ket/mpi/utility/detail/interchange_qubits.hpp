#ifndef KET_MPI_UTILITY_DETAIL_INTERCHANGE_QUBITS_HPP
# define KET_MPI_UTILITY_DETAIL_INTERCHANGE_QUBITS_HPP

# include <cassert>
# include <vector>
# include <iterator>
# include <utility>
# include <type_traits>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/algorithm/swap.hpp>

# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct interchange_qubits
        {
          template <typename LocalState, typename Allocator, typename StateInteger>
          static auto call(
            LocalState&& local_state,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const source_local_first_index, StateInteger const source_local_last_index,
            yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            assert(source_local_last_index >= source_local_first_index);

            using std::begin;
            auto const first = begin(local_state) + data_block_index * data_block_size + source_local_first_index;
            auto const last = begin(local_state) + data_block_index * data_block_size + source_local_last_index;

#ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            auto const new_size = source_local_last_index - source_local_first_index;
            if (new_size > buffer.capacity())
              std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >{}.swap(buffer);
            buffer.resize(new_size);

            using std::end;
            yampi::algorithm::swap(
              yampi::ignore_status,
              yampi::make_buffer(first, last), yampi::make_buffer(begin(buffer), end(buffer)),
              target_rank, communicator, environment);
            std::copy(begin(buffer), end(buffer), first);
#else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            if (buffer.empty())
            {
              auto const new_size = source_local_last_index - source_local_first_index;
              if (new_size > buffer.capacity())
                std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >{}.swap(buffer);
              buffer.resize(new_size);

              using std::end;
              yampi::algorithm::swap(
                yampi::ignore_status,
                yampi::make_buffer(first, last), yampi::make_buffer(begin(buffer), end(buffer)),
                target_rank, communicator, environment);
              std::copy(begin(buffer), end(buffer), first);
            }
            else
            {
              auto const buffer_size = buffer.size();
              auto const num_iterations = (source_local_last_index - source_local_first_index) / buffer_size;

              auto present_first = first;
              for (auto count = decltype(num_iterations){0}; count < num_iterations; ++count)
              {
                using std::end;
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + buffer_size),
                  yampi::make_buffer(begin(buffer), end(buffer)),
                  target_rank, communicator, environment);
                std::copy(begin(buffer), end(buffer), present_first);

                present_first += buffer_size;
              }

              auto const remainder_size = (source_local_last_index - source_local_first_index) % buffer_size;
              if (remainder_size > decltype(remainder_size){0})
              {
                using std::end;
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + remainder_size),
                  yampi::make_buffer(begin(buffer), begin(buffer) + remainder_size),
                  target_rank, communicator, environment);
                std::copy_n(begin(buffer), remainder_size, present_first);
              }
            }
#endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
          }

          template <typename LocalState, typename Allocator, typename StateInteger, typename DerivedDatatype>
          static auto call(
            LocalState&& local_state,
            std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const source_local_first_index, StateInteger const source_local_last_index,
            yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          -> void
          {
            assert(source_local_last_index >= source_local_first_index);

            using std::begin;
            auto const first = begin(local_state) + data_block_index * data_block_size + source_local_first_index;
            auto const last = begin(local_state) + data_block_index * data_block_size + source_local_last_index;

#ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            buffer.resize(source_local_last_index - source_local_first_index);
            using std::end;
            yampi::algorithm::swap(
              yampi::ignore_status,
              yampi::make_buffer(first, last, datatype),
              yampi::make_buffer(begin(buffer), end(buffer), datatype),
              target_rank, communicator, environment);
            std::copy(begin(buffer), end(buffer), first);
#else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            if (buffer.empty())
            {
              buffer.resize(source_local_last_index - source_local_first_index);
              using std::end;
              yampi::algorithm::swap(
                yampi::ignore_status,
                yampi::make_buffer(first, last, datatype),
                yampi::make_buffer(begin(buffer), end(buffer), datatype),
                target_rank, communicator, environment);
              std::copy(begin(buffer), end(buffer), first);
            }
            else
            {
              auto const buffer_size = buffer.size();
              auto const num_iterations = (source_local_last_index - source_local_first_index) / buffer_size;

              auto present_first = first;
              for (auto count = decltype(num_iterations){0}; count < num_iterations; ++count)
              {
                using std::end;
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + buffer_size, datatype),
                  yampi::make_buffer(begin(buffer), end(buffer), datatype),
                  target_rank, communicator, environment);
                std::copy(begin(buffer), end(buffer), present_first);

                present_first += buffer_size;
              }

              auto const remainder_size = (source_local_last_index - source_local_first_index) % buffer_size;
              if (remainder_size > decltype(remainder_size){0})
              {
                using std::end;
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + remainder_size, datatype),
                  yampi::make_buffer(begin(buffer), begin(buffer) + remainder_size, datatype),
                  target_rank, communicator, environment);
                std::copy_n(begin(buffer), remainder_size, present_first);
              }
            }
#endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
          }
        }; // struct interchange_qubits<LocalState_>
      } // namespace dispatch

      namespace detail
      {
        template <typename LocalState, typename Allocator, typename StateInteger>
        inline auto interchange_qubits(
          LocalState&& local_state,
          std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index, StateInteger const source_local_last_index,
          yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          using interchange_qubits_
            = ::ket::mpi::utility::dispatch::interchange_qubits<std::remove_cv_t<std::remove_reference_t<LocalState>>>;
          interchange_qubits_::call(
            std::forward<LocalState>(local_state), buffer,
            data_block_index, data_block_size,
            source_local_first_index, source_local_last_index,
            target_rank, communicator, environment);
        }

        template <typename LocalState, typename Allocator, typename StateInteger, typename DerivedDatatype>
        inline auto interchange_qubits(
          LocalState&& local_state,
          std::vector< ::ket::utility::meta::range_value_t<LocalState>, Allocator >& buffer,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index, StateInteger const source_local_last_index,
          yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> void
        {
          using interchange_qubits_
            = ::ket::mpi::utility::dispatch::interchange_qubits<std::remove_cv_t<std::remove_reference_t<LocalState>>>;
          interchange_qubits_::call(
            std::forward<LocalState>(local_state), buffer,
            data_block_index, data_block_size,
            source_local_first_index, source_local_last_index,
            datatype, target_rank, communicator, environment);
        }
      } // namespace detail
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_DETAIL_INTERCHANGE_QUBITS_HPP
