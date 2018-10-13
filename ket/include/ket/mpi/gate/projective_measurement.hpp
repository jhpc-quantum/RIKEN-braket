#ifndef KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP
# define KET_MPI_GATE_PROJECTIVE_MEASUREMENT_HPP

# include <boost/config.hpp>

# include <cmath>
# include <complex>
# include <ios>
# include <sstream>
# include <vector>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# include <utility>
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else // BOOST_NO_CXX11_ADDRESSOF
#   include <boost/core/addressof.hpp>
# endif // BOOST_NO_CXX11_ADDRESSOF

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/broadcast.hpp>

# include <ket/qubit.hpp>
# include <ket/qubit_io.hpp>
# include <ket/gate/projective_measurement.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/projective_measurement.hpp>
# include <ket/mpi/page/is_on_page.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define KET_addressof std::addressof
# else
#   define KET_addressof boost::addressof
# endif


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
      inline KET_GATE_OUTCOME_TYPE projective_measurement(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const& complex_datatype,
        yampi::datatype const& real_pair_datatype,
        yampi::rank const root,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Measurement ", std::ios_base::ate);
        output_string_stream << qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        KET_array<qubit_type, 1u> qubits = { qubit };
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation,
          buffer, complex_datatype, communicator, environment);

        bool const is_qubit_on_page
          = ::ket::mpi::page::is_on_page(qubit, local_state, permutation);

        typedef typename boost::range_value<RandomAccessRange>::type complex_type;
        typedef typename ::ket::utility::meta::real_of<complex_type>::type real_type;
        std::pair<real_type, real_type> zero_one_probabilities
          = is_qubit_on_page
            ? ::ket::mpi::gate::page::zero_one_probabilities(
                mpi_policy, parallel_policy, local_state, qubit, permutation)
            : ::ket::gate::projective_measurement_detail::zero_one_probabilities(
                parallel_policy, boost::begin(local_state), boost::end(local_state), qubit);

        yampi::all_reduce(
          communicator, environment,
          yampi::make_buffer(zero_one_probabilities, real_pair_datatype),
          KET_addressof(zero_one_probabilities), yampi::binary_operation(yampi::plus_t()));
        real_type const total_probability = zero_one_probabilities.first + zero_one_probabilities.second;

        int zero_or_one
          = ::ket::utility::positive_random_value_upto(total_probability, random_number_generator)
              < zero_one_probabilities.first
            ? 0 : 1;

        yampi::broadcast(communicator, root).call(
          environment,
          yampi::make_buffer(zero_or_one, yampi::datatype(yampi::int_datatype_t())));

        if (zero_or_one == 0)
        {
          if (is_qubit_on_page)
            ::ket::mpi::gate::page::change_state_after_measuring_zero(
              mpi_policy, parallel_policy, local_state, qubit, zero_one_probabilities.first, permutation);
          else
            ::ket::gate::projective_measurement_detail::change_state_after_measuring_zero(
              parallel_policy,
              boost::begin(local_state), boost::end(local_state), qubit, zero_one_probabilities.first);

          return KET_GATE_OUTCOME_VALUE(zero);
        }

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::change_state_after_measuring_one(
            mpi_policy, parallel_policy,
            local_state, qubit, zero_one_probabilities.second, permutation);
        else
          ::ket::gate::projective_measurement_detail::change_state_after_measuring_one(
            parallel_policy,
            boost::begin(local_state), boost::end(local_state), qubit, zero_one_probabilities.second);

        return KET_GATE_OUTCOME_VALUE(one);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename RandomNumberGenerator,
        typename Allocator, typename BufferAllocator>
      inline KET_GATE_OUTCOME_TYPE projective_measurement(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const& complex_datatype,
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
      inline KET_GATE_OUTCOME_TYPE projective_measurement(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        RandomNumberGenerator& random_number_generator,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const& complex_datatype,
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


# undef KET_addressof
# undef KET_array

#endif

