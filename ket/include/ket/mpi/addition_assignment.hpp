#ifndef KET_MPI_ADDITION_ASSIGNMENT_HPP
# define KET_MPI_ADDITION_ASSIGNMENT_HPP

# include <cassert>
# include <cstddef>
# include <iterator>
# include <type_traits>
# include <vector>

# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>

# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/swapped_fourier_transform.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/controlled_phase_shift.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
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
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename Iterator1, typename Iterator2, typename PhaseCoefficientsAllocator>
      inline auto do_addition_assignment(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first, BitInteger const register_size,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
      -> void
      {
        for (auto phase_exponent = BitInteger{1u}; phase_exponent <= register_size; ++phase_exponent)
        {
          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = BitInteger{0u};
               control_bit_index <= register_size - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index = control_bit_index + (phase_exponent - BitInteger{1u});

            ::ket::mpi::gate::controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              phase_coefficient, lhs_qubits_first[target_bit_index], ::ket::make_control(rhs_qubits_first[control_bit_index]));
          }
        }
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename Iterator1, typename Iterator2, typename PhaseCoefficientsAllocator>
      inline auto do_addition_assignment(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first, BitInteger const register_size,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
      -> void
      {
        for (auto phase_exponent = BitInteger{1u}; phase_exponent <= register_size; ++phase_exponent)
        {
          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = BitInteger{0u};
               control_bit_index <= register_size - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index = control_bit_index + (phase_exponent - BitInteger{1u});

            ::ket::mpi::gate::controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              phase_coefficient, lhs_qubits_first[target_bit_index], ::ket::make_control(rhs_qubits_first[control_bit_index]));
          }
        }
      }
    } // namespace addition_assignment_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
        "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Addition", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<BitInteger>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, phase_coefficients);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_addition_assignment(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          begin(lhs_qubits), begin(rhs_qubits), register_size, phase_coefficients);

      ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment
        lhs_qubits, phase_coefficients);

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
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
      RandomAccessRange&>
    addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Addition", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<BitInteger>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, phase_coefficients);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_addition_assignment(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          begin(lhs_qubits), begin(rhs_qubits), register_size, phase_coefficients,

      ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, phase_coefficients);

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
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
    addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
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
    addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return addition_assignment(
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
    addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return addition_assignment(
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
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return addition_assignment(
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
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      static_assert(
        std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
        "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, permutation);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      static_assert(
        std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
        "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
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
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, permutation);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return addition_assignment(
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
    addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return addition_assignment(
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
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return addition_assignment(
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
      RandomAccessRange&>
    addition_assignment(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }


    namespace addition_assignment_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename Iterator1, typename Iterator2, typename PhaseCoefficientsAllocator>
      inline auto do_adj_addition_assignment(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first, BitInteger const register_size,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
      -> void
      {
        for (auto index = BitInteger{0u}; index < register_size; ++index)
        {
          auto const phase_exponent = register_size - index;

          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = BitInteger{0u};
               control_bit_index <= register_size - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index
              = control_bit_index + (phase_exponent - BitInteger{1u});

            ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment);
              phase_coefficient, lhs_qubits_first[target_bit_index], ::ket::make_control(rhs_qubits_first[control_bit_index]));
          }
        }
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename Iterator1, typename Iterator2, typename PhaseCoefficientsAllocator>
      inline auto do_adj_addition_assignment(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Iterator1 const lhs_qubits_first, Iterator2 const rhs_qubits_first, BitInteger const register_size,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
      -> void
      {
        for (auto index = BitInteger{0u}; index < register_size; ++index)
        {
          auto const phase_exponent = register_size - index;

          auto const phase_coefficient = phase_coefficients[phase_exponent];

          for (auto control_bit_index = BitInteger{0u};
               control_bit_index <= register_size - phase_exponent; ++control_bit_index)
          {
            auto const target_bit_index
              = control_bit_index + (phase_exponent - BitInteger{1u});

            ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment);
              phase_coefficient, lhs_qubits_first[target_bit_index], ::ket::make_control(rhs_qubits_first[control_bit_index]));
          }
        }
      }
    } // namespace addition_assignment_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
        "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Addition)", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<BitInteger>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, phase_coefficients);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_adj_addition_assignment(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          begin(lhs_qubits), begin(rhs_qubits), register_size, phase_coefficients);

      ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, phase_coefficients);

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
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
      RandomAccessRange&>
    adj_addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      static_assert(
        std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
        "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Addition)", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<BitInteger>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, phase_coefficients);

      for (auto const& rhs_qubits: rhs_qubits_range)
        ::ket::mpi::addition_assignment_detail::do_adj_addition_assignment(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          begin(lhs_qubits), begin(rhs_qubits), register_size, phase_coefficients);

      ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, phase_coefficients);

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
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
    adj_addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_addition_assignment(
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
    adj_addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_addition_assignment(
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
    adj_addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_addition_assignment(
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
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_addition_assignment(
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
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_addition_assignment(
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
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename QubitsRange>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      static_assert(
        std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
        "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
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
      RandomAccessRange&>
    adj_addition_assignment(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      static_assert(
        std::is_same< ::ket::utility::meta::range_value_t<Qubits>, ::ket::utility::meta::range_value_t< ::ket::utility::meta::range_value_t<QubitsRange> > >::value,
        "Qubits' value_type and QubitsRange's value_type's value_type should be the same");

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(lhs_qubits), end(lhs_qubits)));
      assert(
        std::all_of(
          begin(rhs_qubits_range), end(rhs_qubits_range),
          [register_size](::ket::utility::meta::range_value_t<QubitsRange> const& rhs_qubits)
          { return register_size == static_cast<bit_integer_type>(std::distance(begin(rhs_qubits), end(rhs_qubits))); }));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return adj_addition_assignment(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
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
    adj_addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_addition_assignment(
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_addition_assignment(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_addition_assignment(
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
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, lhs_qubits, rhs_qubits_range,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename QubitsRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    adj_addition_assignment(
      ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& lhs_qubits, QubitsRange const& rhs_qubits_range)
    {
      return ::ket::mpi::adj_addition_assignment(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        lhs_qubits, rhs_qubits_range);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_ADDITION_ASSIGNMENT_HPP
