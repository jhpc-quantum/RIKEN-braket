#ifndef KET_MPI_SWAPPED_FOURIER_TRANSFORM_HPP
# define KET_MPI_SWAPPED_FOURIER_TRANSFORM_HPP

# include <cassert>
# include <cstddef>
# include <type_traits>
# include <vector>

# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/meta/const_iterator_of.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/hadamard.hpp>
# include <ket/mpi/gate/controlled_phase_shift.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
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
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Fourier", environment};

      auto const num_qubits = boost::size(qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      auto const qubits_first = ::ket::utility::begin(qubits);

      for (auto index = decltype(num_qubits){0u}; index < num_qubits; ++index)
      {
        auto target_bit = num_qubits - index - decltype(num_qubits){1u};

        using ::ket::mpi::gate::hadamard;
        hadamard(
          mpi_policy, parallel_policy,
          local_state, qubits_first[target_bit], permutation,
          buffer, communicator, environment);

        for (auto phase_exponent = decltype(num_qubits){2u};
             phase_exponent <= num_qubits - index; ++phase_exponent)
        {
          auto const control_bit = target_bit - (phase_exponent - decltype(phase_exponent){1u});

          using ::ket::mpi::gate::controlled_phase_shift_coeff;
          controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficients[phase_exponent],
            qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]),
            permutation, buffer, communicator, environment);
        }
      }

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Fourier", environment};

      auto const num_qubits = boost::size(qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      auto const qubits_first = ::ket::utility::begin(qubits);

      for (auto index = decltype(num_qubits){0u}; index < num_qubits; ++index)
      {
        auto target_bit = num_qubits - index - decltype(num_qubits){1u};

        using ::ket::mpi::gate::hadamard;
        hadamard(
          mpi_policy, parallel_policy,
          local_state, qubits_first[target_bit], permutation,
          buffer, datatype, communicator, environment);

        for (auto phase_exponent = decltype(num_qubits){2u};
             phase_exponent <= num_qubits - index; ++phase_exponent)
        {
          auto const control_bit = target_bit - (phase_exponent - decltype(phase_exponent){1u});

          using ::ket::mpi::gate::controlled_phase_shift_coeff;
          controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficients[phase_exponent],
            qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]),
            permutation, buffer, datatype, communicator, environment);
        }
      }

      return local_state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return swapped_fourier_transform(
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return swapped_fourier_transform(
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return swapped_fourier_transform(
        parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
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
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return swapped_fourier_transform(
        parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }



    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Fourier)", environment};

      auto const num_qubits = boost::size(qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      auto const qubits_first = ::ket::utility::begin(qubits);

      for (auto target_bit = decltype(num_qubits){0u}; target_bit < num_qubits; ++target_bit)
      {
        for (auto index = decltype(target_bit){0u}; index < target_bit; ++index)
        {
          auto const phase_exponent = decltype(target_bit){1u} + target_bit - index;
          auto const control_bit = target_bit - (phase_exponent - decltype(phase_exponent){1u});

          using ::ket::mpi::gate::adj_controlled_phase_shift_coeff;
          adj_controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficients[phase_exponent], qubits_first[target_bit],
            ::ket::make_control(qubits_first[control_bit]),
            permutation, buffer, communicator, environment);
        }

        using ::ket::mpi::gate::adj_hadamard;
        adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, qubits_first[target_bit], permutation,
          buffer, communicator, environment);
      }

      return local_state;
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      assert(::ket::utility::ranges::is_unique_if_sorted(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print{"Adj(Fourier)", environment};

      auto const num_qubits = boost::size(qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      auto const qubits_first = ::ket::utility::begin(qubits);

      for (auto target_bit = decltype(num_qubits){0u}; target_bit < num_qubits; ++target_bit)
      {
        for (auto index = decltype(target_bit){0u}; index < target_bit; ++index)
        {
          auto const phase_exponent = decltype(target_bit){1u} + target_bit - index;
          auto const control_bit = target_bit - (phase_exponent - decltype(phase_exponent){1u});

          using ::ket::mpi::gate::adj_controlled_phase_shift_coeff;
          adj_controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficients[phase_exponent], qubits_first[target_bit],
            ::ket::make_control(qubits_first[control_bit]),
            permutation, buffer, datatype, communicator, environment);
        }

        using ::ket::mpi::gate::adj_hadamard;
        adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, qubits_first[target_bit], permutation,
          buffer, datatype, communicator, environment);
      }

      return local_state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
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
      return adj_swapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_swapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_swapped_fourier_transform(
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
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
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_swapped_fourier_transform(
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::communicator const& communicator,
      yampi::environment const& environment)
    {
      using complex_type = typename boost::range_value<RandomAccessRange>::type;
      auto phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_swapped_fourier_transform(
        parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_swapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
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
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_swapped_fourier_transform(
        parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_SWAPPED_FOURIER_TRANSFORM_HPP
