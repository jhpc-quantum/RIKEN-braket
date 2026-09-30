#ifndef KET_MPI_GATE_DETAIL_GLOBAL_PHASE_RUNTIME_HPP
# define KET_MPI_GATE_DETAIL_GLOBAL_PHASE_RUNTIME_HPP

# include <complex>
# include <string>
# include <vector>

# include <yampi/communicator.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/environment.hpp>

# include <ket/gate/global_phase.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/utility/meta/ranges.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace runtime
      {
        namespace ranges
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Complex>
          inline auto global_phase_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient)
          -> RandomAccessRange&
          {
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"GlobalPhase(coeff) "}, phase_coefficient),
              environment};

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, &phase_coefficient](auto const first, auto const last)
              { ::ket::gate::runtime::global_phase_coeff(parallel_policy, first, last, phase_coefficient); });
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
          inline auto global_phase_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const&,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::global_phase_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Complex>
          inline auto adj_global_phase_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient)
          -> RandomAccessRange&
          {
            using std::conj;
            return ::ket::mpi::gate::runtime::ranges::global_phase_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, conj(phase_coefficient));
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
          inline auto adj_global_phase_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient)
          -> RandomAccessRange&
          {
            using std::conj;
            return ::ket::mpi::gate::runtime::ranges::global_phase_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, conj(phase_coefficient));
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Real>
          inline auto global_phase(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase)
          -> RandomAccessRange&
          {
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"GlobalPhase "}, phase),
              environment};

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, phase](auto const first, auto const last)
              { ::ket::gate::runtime::global_phase(parallel_policy, first, last, phase); });
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto global_phase(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const&,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::global_phase(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Real>
          inline auto adj_global_phase(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase)
          -> RandomAccessRange&
          { return ::ket::mpi::gate::runtime::ranges::global_phase(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, -phase); }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto adj_global_phase(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase)
          -> RandomAccessRange&
          { return ::ket::mpi::gate::runtime::ranges::global_phase(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, -phase); }
        } // namespace ranges

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex>
        inline auto global_phase_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::global_phase_coeff(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
        inline auto global_phase_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::global_phase_coeff(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase_coefficient); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto global_phase(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::global_phase(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto global_phase(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::global_phase(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex>
        inline auto adj_global_phase_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_global_phase_coeff(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
        inline auto adj_global_phase_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_global_phase_coeff(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase_coefficient); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto adj_global_phase(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_global_phase(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto adj_global_phase(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_global_phase(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase); }
      } // namespace runtime
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_DETAIL_GLOBAL_PHASE_RUNTIME_HPP
