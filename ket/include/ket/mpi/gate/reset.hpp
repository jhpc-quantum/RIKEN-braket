#ifndef KET_MPI_GATE_RESET_HPP
# define KET_MPI_GATE_RESET_HPP

# include <vector>

# include <yampi/communicator.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/environment.hpp>
# include <yampi/rank.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/projective_measurement.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/gate/pauli_x.hpp>
# include <ket/mpi/gate/projective_measurement.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      // RESET_i
      // RESET_i |psi> measures qubit i and maps either outcome to |0>_i.
      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      inline auto reset(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<
          ::ket::utility::meta::range_value_t<RandomAccessRange>,
          BufferAllocator>& buffer,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        auto const outcome
          = ::ket::mpi::gate::projective_measurement(
              mpi_policy, parallel_policy, local_state, permutation, buffer,
              root, communicator, environment, random_number_generator, qubit);
        if (outcome == ::ket::gate::outcome::one)
          ::ket::mpi::gate::pauli_x(
            mpi_policy, parallel_policy, local_state, permutation, buffer,
            communicator, environment, qubit);
        return local_state;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename DerivedDatatype1, typename DerivedDatatype2,
        typename RandomNumberGenerator>
      inline auto reset(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<
          ::ket::utility::meta::range_value_t<RandomAccessRange>,
          BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype1> const& complex_datatype,
        yampi::datatype_base<DerivedDatatype2> const& real_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        auto const outcome
          = ::ket::mpi::gate::projective_measurement(
              mpi_policy, parallel_policy, local_state, permutation, buffer,
              complex_datatype, real_datatype, root, communicator, environment,
              random_number_generator, qubit);
        if (outcome == ::ket::gate::outcome::one)
          ::ket::mpi::gate::pauli_x(
            mpi_policy, parallel_policy, local_state, permutation, buffer,
            complex_datatype, communicator, environment, qubit);
        return local_state;
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      inline auto reset(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<
          ::ket::utility::meta::range_value_t<RandomAccessRange>,
          BufferAllocator>& buffer,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::reset(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename DerivedDatatype1, typename DerivedDatatype2,
        typename RandomNumberGenerator>
      inline auto reset(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<
          ::ket::utility::meta::range_value_t<RandomAccessRange>,
          BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype1> const& complex_datatype,
        yampi::datatype_base<DerivedDatatype2> const& real_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::reset(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, complex_datatype, real_datatype,
          root, communicator, environment, random_number_generator, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      inline auto reset(
        ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<
          ::ket::utility::meta::range_value_t<RandomAccessRange>,
          BufferAllocator>& buffer,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::reset(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename DerivedDatatype1, typename DerivedDatatype2,
        typename RandomNumberGenerator>
      inline auto reset(
        ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<
          ::ket::utility::meta::range_value_t<RandomAccessRange>,
          BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype1> const& complex_datatype,
        yampi::datatype_base<DerivedDatatype2> const& real_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::reset(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, complex_datatype, real_datatype,
          root, communicator, environment, random_number_generator, qubit);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_RESET_HPP
