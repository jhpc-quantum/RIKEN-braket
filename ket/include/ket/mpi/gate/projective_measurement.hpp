#ifndef KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP
# define KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP

# include <cmath>
# include <complex>
# include <vector>
# include <array>
# include <iterator>
# include <utility>
# include <memory>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/broadcast.hpp>

# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/projective_measurement.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/projective_measurement.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      inline auto projective_measurement(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Measurement "}, qubit), environment};

        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);

        auto const permutated_qubit = permutation[qubit];
        auto const is_qubit_on_page = ::ket::mpi::page::is_on_page(permutated_qubit, local_state);

        using std::begin;
        using std::end;
        auto zero_one_probabilities
          = is_qubit_on_page
            ? ::ket::mpi::gate::page::zero_one_probabilities(parallel_policy, local_state, permutated_qubit)
            : ::ket::gate::projective_measurement_detail::zero_one_probabilities(
                parallel_policy, begin(local_state), end(local_state), permutated_qubit.qubit());

        yampi::all_reduce(
          yampi::make_buffer(zero_one_probabilities.first),
          std::addressof(zero_one_probabilities.first), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);
        yampi::all_reduce(
          yampi::make_buffer(zero_one_probabilities.second),
          std::addressof(zero_one_probabilities.second), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);

        auto zero_or_one = 0;

        if (communicator.rank(environment) == root)
          zero_or_one
            = ::ket::utility::positive_random_value_upto(zero_one_probabilities.first + zero_one_probabilities.second, random_number_generator)
                < zero_one_probabilities.first
              ? 0 : 1;

        yampi::broadcast(yampi::make_buffer(zero_or_one), root, communicator, environment);

        if (zero_or_one == 0)
        {
          if (is_qubit_on_page)
            ::ket::mpi::gate::page::change_state_after_measuring_zero(
              parallel_policy, local_state, permutated_qubit, zero_one_probabilities.first);
          else
            ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
              parallel_policy,
              begin(local_state), end(local_state), permutated_qubit.qubit(), zero_one_probabilities.first);

          return ::ket::gate::outcome::zero;
        }

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::change_state_after_measuring_one(
            parallel_policy, local_state, permutated_qubit, zero_one_probabilities.second);
        else
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
            parallel_policy,
            begin(local_state), end(local_state), permutated_qubit.qubit(), zero_one_probabilities.second);

        return ::ket::gate::outcome::one;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Measurement "}, qubit), environment};

        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);

        auto const permutated_qubit = permutation[qubit];
        auto const is_qubit_on_page = ::ket::mpi::page::is_on_page(permutated_qubit, local_state);

        using std::begin;
        using std::end;
        auto zero_one_probabilities
          = is_qubit_on_page
            ? ::ket::mpi::gate::page::zero_one_probabilities(parallel_policy, local_state, permutated_qubit)
            : ::ket::gate::projective_measurement_detail::zero_one_probabilities(
                parallel_policy, begin(local_state), end(local_state), permutated_qubit.qubit());

        yampi::all_reduce(
          yampi::make_buffer(zero_one_probabilities, real_pair_datatype),
          std::addressof(zero_one_probabilities), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);

        auto zero_or_one = 0;

        if (communicator.rank(environment) == root)
          zero_or_one
            = ::ket::utility::positive_random_value_upto(zero_one_probabilities.first + zero_one_probabilities.second, random_number_generator)
                < zero_one_probabilities.first
              ? 0 : 1;

        yampi::broadcast(yampi::make_buffer(zero_or_one), root, communicator, environment);

        if (zero_or_one == 0)
        {
          if (is_qubit_on_page)
            ::ket::mpi::gate::page::change_state_after_measuring_zero(
              parallel_policy, local_state, permutated_qubit, zero_one_probabilities.first);
          else
            ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
              parallel_policy,
              begin(local_state), end(local_state), permutated_qubit.qubit(), zero_one_probabilities.first);

          return ::ket::gate::outcome::zero;
        }

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::change_state_after_measuring_one(
            parallel_policy, local_state, permutated_qubit, zero_one_probabilities.second);
        else
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
            parallel_policy,
            begin(local_state), end(local_state), permutated_qubit.qubit(), zero_one_probabilities.second);

        return ::ket::gate::outcome::one;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype1, typename DerivedDatatype2, typename RandomNumberGenerator>
      inline auto projective_measurement(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype1> const& complex_datatype, yampi::datatype_base<DerivedDatatype2> const& real_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Measurement "}, qubit), environment};

        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, complex_datatype, communicator, environment, qubit);

        auto const permutated_qubit = permutation[qubit];
        auto const is_qubit_on_page = ::ket::mpi::page::is_on_page(permutated_qubit, local_state);

        using std::begin;
        using std::end;
        auto zero_one_probabilities
          = is_qubit_on_page
            ? ::ket::mpi::gate::page::zero_one_probabilities(parallel_policy, local_state, permutated_qubit)
            : ::ket::gate::projective_measurement_detail::zero_one_probabilities(
                parallel_policy, begin(local_state), end(local_state), permutated_qubit.qubit());

        yampi::all_reduce(
          yampi::make_buffer(zero_one_probabilities.first, real_datatype),
          std::addressof(zero_one_probabilities.first), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);
        yampi::all_reduce(
          yampi::make_buffer(zero_one_probabilities.second, real_datatype),
          std::addressof(zero_one_probabilities.second), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);

        auto zero_or_one = 0;

        if (communicator.rank(environment) == root)
          zero_or_one
            = ::ket::utility::positive_random_value_upto(zero_one_probabilities.first + zero_one_probabilities.second, random_number_generator)
                < zero_one_probabilities.first
              ? 0 : 1;

        yampi::broadcast(yampi::make_buffer(zero_or_one), root, communicator, environment);

        if (zero_or_one == 0)
        {
          if (is_qubit_on_page)
            ::ket::mpi::gate::page::change_state_after_measuring_zero(
              parallel_policy, local_state, permutated_qubit, zero_one_probabilities.first);
          else
            ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
              parallel_policy,
              begin(local_state), end(local_state), permutated_qubit.qubit(), zero_one_probabilities.first);

          return ::ket::gate::outcome::zero;
        }

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::change_state_after_measuring_one(
            parallel_policy, local_state, permutated_qubit, zero_one_probabilities.second);
        else
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
            parallel_policy,
            begin(local_state), end(local_state), permutated_qubit.qubit(), zero_one_probabilities.second);

        return ::ket::gate::outcome::one;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype, yampi::datatype const& real_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto projective_measurement(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto projective_measurement(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto projective_measurement(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto projective_measurement(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      inline auto projective_measurement(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename RandomNumberGenerator>
      inline auto projective_measurement(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype, yampi::datatype const& real_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_datatype, yampi::rank const root,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, RandomNumberGenerator& random_number_generator)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          qubit, random_number_generator);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      inline auto projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename RandomNumberGenerator>
      [[deprecated]] inline auto projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, real_pair_datatype, root, communicator, environment,
          random_number_generator, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename RandomNumberGenerator>
      inline auto projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        RandomNumberGenerator& random_number_generator, ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> ::ket::gate::outcome
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, complex_datatype, real_datatype, root, communicator, environment,
          random_number_generator, qubit);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP
