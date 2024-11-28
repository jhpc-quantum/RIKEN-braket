#ifndef KET_MPI_PAGE_PAGE_SIZE_HPP
# define KET_MPI_PAGE_PAGE_SIZE_HPP

# include <type_traits>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>

# include <ket/mpi/utility/simple_mpi.hpp>

namespace ket
{
  namespace mpi
  {
    namespace page
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct page_size
        {
          template <typename MpiPolicy, typename LocalState>
          static auto call(MpiPolicy const& mpi_policy, LocalState const& local_state, yampi::communicator const& communicator, yampi::environment const& environment)
          { return ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment); }
        }; // struct page_size<LocalState_>
      } // namespace dispatch

      template <typename MpiPolicy, typename LocalState>
      inline auto page_size(MpiPolicy const& mpi_policy, LocalState const& local_state, yampi::communicator const& communicator, yampi::environment const& environment)
      { return ::ket::mpi::page::dispatch::page_size<std::remove_cv_t<std::remove_reference_t<LocalState>>>::call(mpi_policy, local_state, communicator, environment); }
    } // namespace page
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PAGE_PAGE_SIZE_HPP
