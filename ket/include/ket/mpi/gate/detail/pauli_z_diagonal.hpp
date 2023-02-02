#ifndef KET_MPI_GATE_DETAIL_PAULI_Z_DIAGONAL_HPP
# define KET_MPI_GATE_DETAIL_PAULI_Z_DIAGONAL_HPP

# include <boost/config.hpp>

# include <complex>
# include <vector>
# include <iterator>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/pauli_z.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace pauli_z_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename StateInteger>
        struct return_
        {
          template <typename Iterator>
          void operator()(Iterator const, StateInteger const) const { }
        }; // struct return_<StateInteger>

        template <typename StateInteger>
        struct do_pauli_z
        {
          template <typename Iterator>
          void operator()(Iterator const iter, StateInteger const) const
          {
            using complex_type = typename std::iterator_traits<Iterator>::value_type;
            using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
            *iter *= real_type{-1};
          }
        }; // struct do_pauli_z<StateInteger, Complex>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::pauli_z(parallel_policy, local_state, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            [](auto const, StateInteger const) { },
            [](auto const iter, StateInteger const) { *iter *= real_type{-1}; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            ::ket::mpi::gate::pauli_z_detail::return_<StateInteger>{},
            ::ket::mpi::gate::pauli_z_detail::do_pauli_z<StateInteger>{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          return local_state;
        }
      } // namespace pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Z "}, qubit), environment};

        return ::ket::mpi::gate::pauli_z_detail::pauli_z(
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::datatype_base<DerivedDatatype> const&,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Z "}, qubit), environment};

        return ::ket::mpi::gate::pauli_z_detail::pauli_z(
          mpi_policy, parallel_policy, local_state, qubit, permutation, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_z(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_z(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      namespace pauli_z_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using std::conj;
          return ::ket::mpi::gate::pauli_z_detail::pauli_z(
            mpi_policy, parallel_policy,
            local_state, qubit, permutation, communicator, environment);
        }
      } // namespace pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Z) "}, qubit), environment};

        return ::ket::mpi::gate::pauli_z_detail::adj_pauli_z(
          mpi_policy, parallel_policy,
          local_state, qubit, permutation, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::datatype_base<DerivedDatatype> const&,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Z) "}, qubit), environment};

        return ::ket::mpi::gate::pauli_z_detail::adj_pauli_z(
          mpi_policy, parallel_policy,
          local_state, qubit, permutation, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_z(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_z(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, qubit, permutation, buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_PAULI_Z_DIAGONAL_HPP
