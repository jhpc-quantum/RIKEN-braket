#ifndef KET_MPI_GATE_SET_HPP
# define KET_MPI_GATE_SET_HPP

# include <cmath>
# include <iterator>
# include <vector>
# include <array>

# include <boost/math/constants/constants.hpp>

# include <yampi/all_reduce.hpp>
# include <yampi/buffer.hpp>
# include <yampi/binary_operation.hpp>
# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/set.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/set.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      // SET_i
      // SET_1 (a_{0} |0> + a_{1} |1>) = |1>
      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      inline auto set(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Set "}, qubit), environment};

        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);

        auto const permutated_qubit = permutation[qubit];
        auto const is_qubit_on_page = ::ket::mpi::page::is_on_page(permutated_qubit, local_state);

        using std::begin;
        using std::end;
        auto one_probability
          = is_qubit_on_page
            ? ::ket::mpi::gate::page::one_probability(parallel_policy, local_state, permutated_qubit)
            : ::ket::gate::set_detail::one_probability(
                parallel_policy, begin(local_state), end(local_state), permutated_qubit.qubit());

        yampi::all_reduce(
          yampi::make_buffer(one_probability),
          std::addressof(one_probability), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);

        using std::pow;
        using boost::math::constants::half;
        auto multiplier = pow(one_probability, -half<decltype(one_probability)>());

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::do_set(parallel_policy, local_state, multiplier, permutated_qubit);
        else
          ::ket::gate::set_detail::do_set(parallel_policy, begin(local_state), end(local_state), multiplier, permutated_qubit.qubit());

        return local_state;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline auto set(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Set "}, qubit), environment};

        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);

        auto const permutated_qubit = permutation[qubit];
        auto const is_qubit_on_page = ::ket::mpi::page::is_on_page(permutated_qubit, local_state);

        using std::begin;
        using std::end;
        auto one_probability
          = is_qubit_on_page
            ? static_cast<long double>(::ket::mpi::gate::page::one_probability(parallel_policy, local_state, permutated_qubit))
            : static_cast<long double>(::ket::gate::set_detail::one_probability(
                parallel_policy, begin(local_state), end(local_state), permutated_qubit.qubit()));

        yampi::all_reduce(
          yampi::make_buffer(one_probability),
          std::addressof(one_probability), yampi::binary_operation(yampi::plus_t()),
          communicator, environment);

        using real_type = ::ket::utility::meta::real_t<std::remove_cv_t<std::remove_reference_t<RandomAccessRange>>>;
        using std::pow;
        using boost::math::constants::half;
        auto multiplier = static_cast<real_type>(pow(one_probability, -half<long double>()));

        if (is_qubit_on_page)
          ::ket::mpi::gate::page::do_set(parallel_policy, local_state, multiplier, permutated_qubit);
        else
          ::ket::gate::set_detail::do_set(parallel_policy, begin(local_state), end(local_state), multiplier, permutated_qubit.qubit());

        return local_state;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto set(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      { return ::ket::mpi::gate::set(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, qubit); }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto set(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      { return ::ket::mpi::gate::set(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, qubit); }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto set(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto set(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline auto set(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline auto set(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto set(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      [[deprecated]] inline auto set(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
      inline auto set(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline auto set(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::set(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_SET_HPP
