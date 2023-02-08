#ifndef KET_MPI_UTILITY_DETAIL_INTERCHANGE_QUBITS_HPP
# define KET_MPI_UTILITY_DETAIL_INTERCHANGE_QUBITS_HPP

# include <cassert>
# include <vector>
# include <iterator>
# include <utility>
# include <type_traits>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/rank.hpp>
# include <yampi/status.hpp>
# include <yampi/algorithm/swap.hpp>


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
          static void call(
            LocalState&& local_state,
            std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            assert(source_local_last_index >= source_local_first_index);

            auto const first = std::begin(local_state) + data_block_index * data_block_size + source_local_first_index;
            auto const last = std::begin(local_state) + data_block_index * data_block_size + source_local_last_index;

#ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            buffer.resize(source_local_last_index - source_local_first_index);
            yampi::algorithm::swap(
              yampi::ignore_status,
              yampi::make_buffer(first, last),
              yampi::make_buffer(std::begin(buffer), std::end(buffer)),
              target_rank, communicator, environment);
            std::copy(std::begin(buffer), std::end(buffer), first);
#else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            if (buffer.empty())
            {
              buffer.resize(source_local_last_index - source_local_first_index);
              yampi::algorithm::swap(
                yampi::ignore_status,
                yampi::make_buffer(first, last),
                yampi::make_buffer(std::begin(buffer), std::end(buffer)),
                target_rank, communicator, environment);
              std::copy(std::begin(buffer), std::end(buffer), first);
            }
            else
            {
              auto const buffer_size = buffer.size();
              auto const num_iterations = (source_local_last_index - source_local_first_index) / buffer_size;

              auto present_first = first;
              for (auto count = decltype(num_iterations){0}; count < num_iterations; ++count)
              {
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + buffer_size),
                  yampi::make_buffer(std::begin(buffer), std::end(buffer)),
                  target_rank, communicator, environment);
                std::copy(std::begin(buffer), std::end(buffer), present_first);

                present_first += buffer_size;
              }

              auto const remainder_size = (source_local_last_index - source_local_first_index) % buffer_size;
              if (remainder_size > decltype(remainder_size){0})
              {
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + remainder_size),
                  yampi::make_buffer(std::begin(buffer), std::begin(buffer) + remainder_size),
                  target_rank, communicator, environment);
                std::copy(std::begin(buffer), std::begin(buffer) + remainder_size, present_first);
              }
            }
#endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
          }

          template <typename LocalState, typename Allocator, typename StateInteger, typename DerivedDatatype>
          static void call(
            LocalState&& local_state,
            std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer,
            StateInteger const data_block_index, StateInteger const data_block_size,
            StateInteger const source_local_first_index,
            StateInteger const source_local_last_index,
            yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
            yampi::communicator const& communicator, yampi::environment const& environment)
          {
            assert(source_local_last_index >= source_local_first_index);

            auto const first = std::begin(local_state) + data_block_index * data_block_size + source_local_first_index;
            auto const last = std::begin(local_state) + data_block_index * data_block_size + source_local_last_index;

#ifndef BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            buffer.resize(source_local_last_index - source_local_first_index);
            yampi::algorithm::swap(
              yampi::ignore_status,
              yampi::make_buffer(first, last, datatype),
              yampi::make_buffer(std::begin(buffer), std::end(buffer), datatype),
              target_rank, communicator, environment);
            std::copy(std::begin(buffer), std::end(buffer), first);
#else // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
            if (buffer.empty())
            {
              buffer.resize(source_local_last_index - source_local_first_index);
              yampi::algorithm::swap(
                yampi::ignore_status,
                yampi::make_buffer(first, last, datatype),
                yampi::make_buffer(std::begin(buffer), std::end(buffer), datatype),
                target_rank, communicator, environment);
              std::copy(std::begin(buffer), std::end(buffer), first);
            }
            else
            {
              auto const buffer_size = buffer.size();
              auto const num_iterations = (source_local_last_index - source_local_first_index) / buffer_size;

              auto present_first = first;
              for (auto count = decltype(num_iterations){0}; count < num_iterations; ++count)
              {
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + buffer_size, datatype),
                  yampi::make_buffer(std::begin(buffer), std::end(buffer), datatype),
                  target_rank, communicator, environment);
                std::copy(std::begin(buffer), std::end(buffer), present_first);

                present_first += buffer_size;
              }

              auto const remainder_size = (source_local_last_index - source_local_first_index) % buffer_size;
              if (remainder_size > decltype(remainder_size){0})
              {
                yampi::algorithm::swap(
                  yampi::ignore_status,
                  yampi::make_buffer(present_first, present_first + remainder_size, datatype),
                  yampi::make_buffer(std::begin(buffer), std::begin(buffer) + remainder_size, datatype),
                  target_rank, communicator, environment);
                std::copy(std::begin(buffer), std::begin(buffer) + remainder_size, present_first);
              }
            }
#endif // BRAKET_ENABLE_MULTIPLE_USES_OF_BUFFER_FOR_ONE_DATA_TRANSFER_IF_NO_PAGE_EXISTS
          }
        }; // struct interchange_qubits<LocalState_>
      } // namespace dispatch

      namespace detail
      {
        template <typename LocalState, typename Allocator, typename StateInteger>
        inline void interchange_qubits(
          LocalState&& local_state,
          std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using interchange_qubits_
            = ::ket::mpi::utility::dispatch::interchange_qubits<typename std::remove_cv<typename std::remove_reference<LocalState>::type>::type>;
          interchange_qubits_::call(
            std::forward<LocalState>(local_state), buffer,
            data_block_index, data_block_size,
            source_local_first_index, source_local_last_index,
            target_rank, communicator, environment);
        }

        template <typename LocalState, typename Allocator, typename StateInteger, typename DerivedDatatype>
        inline void interchange_qubits(
          LocalState&& local_state,
          std::vector<typename boost::range_value<LocalState>::type, Allocator>& buffer,
          StateInteger const data_block_index, StateInteger const data_block_size,
          StateInteger const source_local_first_index,
          StateInteger const source_local_last_index,
          yampi::datatype_base<DerivedDatatype> const& datatype, yampi::rank const target_rank,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using interchange_qubits_
            = ::ket::mpi::utility::dispatch::interchange_qubits<typename std::remove_cv<typename std::remove_reference<LocalState>::type>::type>;
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
