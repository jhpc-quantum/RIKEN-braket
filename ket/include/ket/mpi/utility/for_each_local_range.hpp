#ifndef KET_MPI_UTILITY_FOR_EACH_LOCAL_RANGE_HPP
# define KET_MPI_UTILITY_FOR_EACH_LOCAL_RANGE_HPP

# include <cstddef>
# include <iterator>
# include <utility>

# include <ket/mpi/utility/simple_mpi.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct for_each_local_range
        {
          template <typename MpiPolicy, typename LocalState, typename StateInteger, typename Function>
          static auto call(
            MpiPolicy const& mpi_policy, LocalState&& local_state,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, Function&& function)
          -> LocalState&&
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

              function(first + data_block_index * data_block_size, first + (data_block_index + 1u) * data_block_size);
            }

            return std::forward<LocalState>(local_state);
          }

          template <typename MpiPolicy, typename LocalState, typename Function>
          static auto call(
            MpiPolicy const& mpi_policy, LocalState&& local_state,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Function&& function)
          -> LocalState&&
          {
            auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, rank_in_unit);
            auto const num_data_blocks = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit);

            using std::begin;
            auto const first = begin(local_state);
            for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
              function(first + data_block_index * data_block_size, first + (data_block_index + 1u) * data_block_size);

            return std::forward<LocalState>(local_state);
          }
        }; // struct for_each_local_range<LocalState_>
      } // namespace dispatch

      template <typename MpiPolicy, typename LocalState, typename StateInteger, typename Function>
      inline auto for_each_local_range(
        MpiPolicy const& mpi_policy, LocalState&& local_state,
        yampi::communicator const& communicator, yampi::environment const& environment,
        StateInteger const unit_control_qubit_mask, Function&& function)
      -> LocalState&&
      {
        return ::ket::mpi::utility::dispatch::for_each_local_range<std::remove_const_t<std::remove_reference_t<LocalState>>>::call(
          mpi_policy, std::forward<LocalState>(local_state), communicator, environment, unit_control_qubit_mask, std::forward<Function>(function));
      }

      template <typename MpiPolicy, typename LocalState, typename Function>
      inline auto for_each_local_range(
        MpiPolicy const& mpi_policy, LocalState&& local_state,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Function&& function)
      -> LocalState&&
      {
        return ::ket::mpi::utility::dispatch::for_each_local_range<std::remove_const_t<std::remove_reference_t<LocalState>>>::call(
          mpi_policy, std::forward<LocalState>(local_state), communicator, environment, std::forward<Function>(function));
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_FOR_EACH_LOCAL_RANGE_HPP
