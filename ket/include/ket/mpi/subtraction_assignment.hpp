#ifndef KET_MPI_SUBTRACTION_ASSIGNMENT_HPP
# define KET_MPI_SUBTRACTION_ASSIGNMENT_HPP

# include <boost/config.hpp>

# include <vector>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/utility/enable_if.hpp>
# endif

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/mpi/addition_assignment.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_enable_if std::enable_if
# else
#   define KET_enable_if boost::enable_if_c
# endif


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
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::adj_addition_assignment;
      return adj_addition_assignment(
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
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::adj_addition_assignment;
      return adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
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
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::adj_addition_assignment;
      return adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::adj_addition_assignment;
      return adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }



    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::addition_assignment;
      return addition_assignment(
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
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::addition_assignment;
      return addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
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
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::addition_assignment;
      return addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using ::ket::mpi::addition_assignment;
      return addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_subtraction_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      return adj_subtraction_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, permutation,
        buffer, datatype, communicator, environment);
    }
  } // namespace mpi
} // namespace ket


# undef KET_enable_if

#endif

