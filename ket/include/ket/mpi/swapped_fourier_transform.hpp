#ifndef KET_MPI_SWAPPED_FOURIER_TRANSFORM_HPP
# define KET_MPI_SWAPPED_FOURIER_TRANSFORM_HPP

# include <cassert>
# include <cstddef>
# include <iterator>
# include <type_traits>
# include <vector>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/hadamard.hpp>
# include <ket/mpi/gate/controlled_phase_shift.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/is_unique_if_sorted.hpp>
# endif


namespace ket
{
  namespace mpi
  {
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Fourier", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));
      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      auto const qubits_first = begin(qubits);

      for (auto index = BitInteger{0u}; index < register_size; ++index)
      {
        auto target_bit = register_size - index - BitInteger{1u};

        ::ket::mpi::gate::hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          qubits_first[target_bit]);

        for (auto phase_exponent = BitInteger{2u};
             phase_exponent <= register_size - index; ++phase_exponent)
        {
          auto const control_bit = target_bit - (phase_exponent - BitInteger{1u});

          ::ket::mpi::gate::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment,
            phase_coefficients[phase_exponent],
            qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]));
        }
      }

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubit, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Fourier", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));
      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      auto const qubits_first = begin(qubits);

      for (auto index = BitInteger{0u}; index < register_size; ++index)
      {
        auto target_bit = register_size - index - BitInteger{1u};

        ::ket::mpi::gate::hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          qubits_first[target_bit]);

        for (auto phase_exponent = BitInteger{2u};
             phase_exponent <= register_size - index; ++phase_exponent)
        {
          auto const control_bit = target_bit - (phase_exponent - BitInteger{1u});

          ::ket::mpi::gate::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            phase_coefficients[phase_exponent],
            qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]));
        }
      }

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubit, phase_coefficients);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utilty::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utilty::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits, phase_coefficients);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        loal_state, permutation, buffer, communicator, environment,
        qubits);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        mpi_policy, parallel_policy,
        loal_state, permutation, buffer, datatype, communicator, environment,
        qubits);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits,
        permutation, buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        qubits);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits,
        permutation, buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubits);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits);
    }



    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Fourier)", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));
      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      auto const qubits_first = begin(qubits);

      for (auto target_bit = BitInteger{0u}; target_bit < register_size; ++target_bit)
      {
        for (auto index = BitInteger{0u}; index < target_bit; ++index)
        {
          auto const phase_exponent = BitInteger{1u} + target_bit - index;
          auto const control_bit = target_bit - (phase_exponent - BitInteger{1u});

          ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment,
            phase_coefficients[phase_exponent],
            qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]));
        }

        ::ket::mpi::gate::adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          qubits_first[target_bit]);
      }

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubit, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Fourier)", environment};

      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));
      ::ket::utility::generate_phase_coefficients(phase_coefficients, register_size);

      auto const qubits_first = begin(qubits);

      for (auto target_bit = BitInteger{0u}; target_bit < register_size; ++target_bit)
      {
        for (auto index = BitInteger{0u}; index < target_bit; ++index)
        {
          auto const phase_exponent = BitInteger{1u} + target_bit - index;
          auto const control_bit = target_bit - (phase_exponent - BitInteger{1u});

          ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            phase_coefficients[phase_exponent],
            qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]));
        }

        ::ket::mpi::gate::adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          qubits_first[target_bit]);
      }

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubit, phase_coefficients);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits, typename PhaseCoefficientsAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, PhaseCoefficientsAllocator >& phase_coefficients)
    {
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits, phase_coefficients);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        loal_state, permutation, buffer, communicator, environment,
        qubits);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      using std::begin;
      using std::end;
      auto const register_size = static_cast<BitInteger>(std::distance(begin(qubits), end(qubits)));

      using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
      auto phase_coefficients = ::ket::utility::generate_phase_coefficients<complex_type>(register_size);

      return ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits, phase_coefficients);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        loal_state, permutation, buffer, datatype, communicator, environment,
        qubits);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits,
        permutation, buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, communicator, environment,
        qubits);
    }

    template <
      typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits,
        permutation, buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype>
    [[deprecated]] inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, qubits,
        permutation, buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      typename Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange& >
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, communicator, environment,
        qubits);
    }

    template <
      typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype,
      typename Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Qubits const& qubits)
    {
      return ::ket::mpi::adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, communicator, environment,
        qubits);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_SWAPPED_FOURIER_TRANSFORM_HPP
