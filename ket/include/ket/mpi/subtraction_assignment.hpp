#ifndef KET_MPI_SUBTRACTION_ASSIGNMENT_HPP
# define KET_MPI_SUBTRACTION_ASSIGNMENT_HPP

# include <vector>
# include <type_traits>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/addition_assignment.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>


namespace ket
{
  namespace mpi
  {
    // lhs -= rhs
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation< StateInteger, BitInteger, Allocator >& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation< StateInteger, BitInteger, Allocator >& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }



    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_SUBTRACTION_ASSIGNMENT_HPP
