#ifndef KET_MPI_PAGE_IS_ON_PAGE_HPP
# define KET_MPI_PAGE_IS_ON_PAGE_HPP

# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/mpi/permutated.hpp>


namespace ket
{
  namespace mpi
  {
    namespace page
    {
      namespace dispatch
      {
        template <typename LocalState_>
        struct is_on_page
        {
          template <typename Qubit, typename LocalState>
          static constexpr auto call(::ket::mpi::permutated<Qubit> const, LocalState const&) -> bool
          { return false; }
        }; // struct is_on_page<LocalState_>
      } // namespace dispatch

      template <typename Qubit, typename LocalState>
      inline constexpr auto is_on_page(::ket::mpi::permutated<Qubit> const permutated_qubit, LocalState const& local_state) -> bool
      { return ::ket::mpi::page::dispatch::is_on_page<std::remove_cv_t<std::remove_reference_t<LocalState>>>::call(permutated_qubit, local_state); }
    } // namespace page
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_PAGE_IS_ON_PAGE_HPP
