#ifndef KET_MPI_ADDITION_ASSIGNMENT_HPP
# define KET_MPI_ADDITION_ASSIGNMENT_HPP

# include <cassert>
# include <cstddef>
# include <type_traits>
# include <vector>

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>

# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/begin.hpp>
# include <ket/mpi/swapped_fourier_transform.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/controlled_phase_shift.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>


namespace ket
{
  namespace mpi
  {
    // lhs += rhs
    namespace addition_assignment_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Iterator1, typename Iterator2,
        typename PhaseCoefficientsAllocator,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      inline void do_addition_assignment(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first,
        std::size_t const num_qubits,
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
        for (auto phase_exponent = std::size_t{1u};
             phase_exponent <= num_qubits; ++phase_exponent)
        {
          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = std::size_t{0u};
               control_bit_index <= num_qubits - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index
              = control_bit_index + (phase_exponent - std::size_t{1u});

            using ::ket::mpi::gate::controlled_phase_shift_coeff;
            controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient,
              lhs_qubits_first[target_bit_index],
              ::ket::make_control(rhs_qubits_first[control_bit_index]),
              permutation, buffer, communicator, environment);
          }
        }
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Iterator1, typename Iterator2,
        typename PhaseCoefficientsAllocator,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype>
      inline void do_addition_assignment(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first,
        std::size_t const num_qubits,
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
        for (auto phase_exponent = std::size_t{1u};
             phase_exponent <= num_qubits; ++phase_exponent)
        {
          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = std::size_t{0u};
               control_bit_index <= num_qubits - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index
              = control_bit_index + (phase_exponent - std::size_t{1u});

            using ::ket::mpi::gate::controlled_phase_shift_coeff;
            controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient,
              lhs_qubits_first[target_bit_index],
              ::ket::make_control(rhs_qubits_first[control_bit_index]),
              permutation, buffer, datatype, communicator, environment);
          }
        }
      }
    } // namespace addition_assignment_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
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
      static_assert(
        std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(
        std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Addition", environment};

      auto const num_qubits = boost::size(lhs_qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      using ::ket::mpi::swapped_fourier_transform;
      swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, communicator, environment);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_addition_assignment(
          mpi_policy, parallel_policy, local_state,
          ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients,
          permutation, buffer, communicator, environment);

      using ::ket::mpi::adj_swapped_fourier_transform;
      adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, communicator, environment);

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
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
      static_assert(
        std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(
        std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Addition", environment};

      auto const num_qubits = boost::size(lhs_qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      using ::ket::mpi::swapped_fourier_transform;
      swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_addition_assignment(
          mpi_policy, parallel_policy, local_state,
          ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients,
          permutation, buffer, datatype, communicator, environment);

      using ::ket::mpi::adj_swapped_fourier_transform;
      adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);

      return local_state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    addition_assignment(
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
      return addition_assignment(
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
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    addition_assignment(
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
      return addition_assignment(
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
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
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
      return addition_assignment(
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
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
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
      return addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
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
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
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
      uisng complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    addition_assignment(
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
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    namespace addition_assignment_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Iterator1, typename Iterator2,
        typename PhaseCoefficientsAllocator,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      inline void do_adj_addition_assignment(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first,
        std::size_t const num_qubits,
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
        for (auto index = std::size_t{0u}; index < num_qubits; ++index)
        {
          auto const phase_exponent = num_qubits - index;

          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = std::size_t{0u};
               control_bit_index <= num_qubits - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index
              = control_bit_index + (phase_exponent - std::size_t{1u});

            using ::ket::mpi::gate::adj_controlled_phase_shift_coeff;
            adj_controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient,
              lhs_qubits_first[target_bit_index],
              ::ket::make_control(rhs_qubits_first[control_bit_index]),
              permutation, buffer, communicator, environment);
          }
        }
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Iterator1, typename Iterator2,
        typename PhaseCoefficientsAllocator,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype>
      inline void do_adj_addition_assignment(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first,
        std::size_t const num_qubits,
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
        for (auto index = std::size_t{0u}; index < num_qubits; ++index)
        {
          auto const phase_exponent = num_qubits - index;

          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = std::size_t{0u};
               control_bit_index <= num_qubits - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index
              = control_bit_index + (phase_exponent - std::size_t{1u});

            using ::ket::mpi::gate::adj_controlled_phase_shift_coeff;
            adj_controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient,
              lhs_qubits_first[target_bit_index],
              ::ket::make_control(rhs_qubits_first[control_bit_index]),
              permutation, buffer, datatype, communicator, environment);
          }
        }
      }
    } // namespace addition_assignment_detail


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      static_assert(
        std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(
        std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Addition)", environment};

      auto const num_qubits = boost::size(lhs_qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      using ::ket::mpi::swapped_fourier_transform;
      swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, communicator, environment);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_adj_addition_assignment(
          mpi_policy, parallel_policy, local_state,
          ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients,
          permutation, buffer, communicator, environment);

      using ::ket::mpi::adj_swapped_fourier_transform;
      adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, communicator, environment);

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      static_assert(
        std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(
        std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Addition)", environment};

      auto const num_qubits = boost::size(lhs_qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      using ::ket::mpi::swapped_fourier_transform;
      swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_adj_addition_assignment(
          mpi_policy, parallel_policy, local_state,
          ::ket::utility::begin(lhs_qubits), ::ket::utility::begin(rhs_qubits), num_qubits, phase_coefficients,
          permutation, buffer, datatype, communicator, environment);

      using ::ket::mpi::adj_swapped_fourier_transform;
      adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);

      return local_state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      return adj_addition_assignment(
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
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      return adj_addition_assignment(
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
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      return adj_addition_assignment(
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
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_addition_assignment(
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
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(lhs_qubits));

      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_general_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_ADDITION_ASSIGNMENT_HPP
