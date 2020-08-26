#ifndef KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP
# define KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP

# include <cmath>
# include <complex>
# include <ios>
# include <sstream>
# include <vector>
# include <array>
# include <utility>
# include <memory>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/broadcast.hpp>

# include <ket/qubit.hpp>
# include <ket/qubit_io.hpp>
# include <ket/gate/projective_measurement.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/projective_measurement.hpp>
# include <ket/mpi/page/is_on_page.hpp>


// TODO: implement vector-support (KET_PREFER_POINTER_TO_VECTOR_ITERATOR)
// avoid direct calling ket::gate::projective_measurement_detail::*
namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator>
      inline ::ket::gate::outcome projective_measurement(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        auto output_string_stream = std::ostringstream{"Measurement ", std::ios_base::ate};
        output_string_stream << qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print{output_string_stream.str(), environment};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto qubits = std::array<qubit_type, 1u>{qubit};
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation, buffer, communicator, environment);

        auto const is_qubit_on_page
          = ::ket::mpi::page::is_on_page(qubit, local_state, permutation);

        auto zero_one_probabilities
          = is_qubit_on_page
            ? ::ket::mpi::gate::page::zero_one_probabilities(
                mpi_policy, parallel_policy, local_state, qubit, permutation)
            : ::ket::gate::projective_measurement_detail::zero_one_probabilities(
                parallel_policy, ::ket::utility::begin(local_state), ::ket::utility::end(local_state), qubit);

        yampi::all_reduce(
          yampi::make_buffer(zero_one_probabilities, real_pair_datatype),
          std::addressof(zero_one_probabilities), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);
        auto const total_probability = zero_one_probabilities.first + zero_one_probabilities.second;

        auto zero_or_one
          = ::ket::utility::positive_random_value_upto(total_probability, random_number_generator)
              < zero_one_probabilities.first
            ? 0 : 1;

        yampi::broadcast(root, communicator).call(yampi::make_buffer(zero_or_one), environment);

        if (zero_or_one == 0)
        {
          if (is_qubit_on_page)
            ::ket::mpi::gate::page::change_state_after_measuring_zero(
              mpi_policy, parallel_policy, local_state, qubit, zero_one_probabilities.first, permutation);
          else
            ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
              parallel_policy,
              ::ket::utility::begin(local_state), ::ket::utility::end(local_state), qubit, zero_one_probabilities.first);

          return ::ket::gate::outcome::zero;
        }

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::change_state_after_measuring_one(
            mpi_policy, parallel_policy,
            local_state, qubit, zero_one_probabilities.second, permutation);
        else
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
            parallel_policy,
            ::ket::utility::begin(local_state), ::ket::utility::end(local_state), qubit, zero_one_probabilities.second);

        return ::ket::gate::outcome::one;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline ::ket::gate::outcome projective_measurement(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        auto output_string_stream = std::ostringstream{"Measurement ", std::ios_base::ate};
        output_string_stream << qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print{output_string_stream.str(), environment};

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        auto qubits = std::array<qubit_type, 1u>{qubit};
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation, buffer, complex_datatype, communicator, environment);

        auto const is_qubit_on_page
          = ::ket::mpi::page::is_on_page(qubit, local_state, permutation);

        auto zero_one_probabilities
          = is_qubit_on_page
            ? ::ket::mpi::gate::page::zero_one_probabilities(
                mpi_policy, parallel_policy, local_state, qubit, permutation)
            : ::ket::gate::projective_measurement_detail::zero_one_probabilities(
                parallel_policy, ::ket::utility::begin(local_state), ::ket::utility::end(local_state), qubit);

        yampi::all_reduce(
          yampi::make_buffer(zero_one_probabilities, real_pair_datatype),
          std::addressof(zero_one_probabilities), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);
        auto const total_probability = zero_one_probabilities.first + zero_one_probabilities.second;

        auto zero_or_one
          = ::ket::utility::positive_random_value_upto(total_probability, random_number_generator)
              < zero_one_probabilities.first
            ? 0 : 1;

        yampi::broadcast(root, communicator).call(yampi::make_buffer(zero_or_one), environment);

        if (zero_or_one == 0)
        {
          if (is_qubit_on_page)
            ::ket::mpi::gate::page::change_state_after_measuring_zero(
              mpi_policy, parallel_policy, local_state, qubit, zero_one_probabilities.first, permutation);
          else
            ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
              parallel_policy,
              ::ket::utility::begin(local_state), ::ket::utility::end(local_state), qubit, zero_one_probabilities.first);

          return ::ket::gate::outcome::zero;
        }

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::change_state_after_measuring_one(
            mpi_policy, parallel_policy,
            local_state, qubit, zero_one_probabilities.second, permutation);
        else
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
            parallel_policy,
            ::ket::utility::begin(local_state), ::ket::utility::end(local_state), qubit, zero_one_probabilities.second);

        return ::ket::gate::outcome::one;
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator>
      inline ::ket::gate::outcome projective_measurement(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, random_number_generator, permutation,
          buffer, real_pair_datatype, root, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline ::ket::gate::outcome projective_measurement(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, random_number_generator, permutation,
          buffer, complex_datatype, real_pair_datatype, root, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator>
      inline ::ket::gate::outcome projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, qubit, random_number_generator, permutation,
          buffer, real_pair_datatype, root, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline ::ket::gate::outcome projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& complex_datatype,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::projective_measurement(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, qubit, random_number_generator, permutation,
          buffer, complex_datatype, real_pair_datatype, root, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP
