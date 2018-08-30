#ifndef KET_MPI_GATE_CONTROLLED_NOT_HPP
# define KET_MPI_GATE_CONTROLLED_NOT_HPP

# include <boost/config.hpp>

# include <vector>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# include <ios>
# include <sstream>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/qubit_io.hpp>
# include <ket/control_io.hpp>
# include <ket/gate/controlled_not.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/controlled_not.hpp>
# include <ket/mpi/page/is_on_page.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace controlled_not_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_controlled_not
        {
          ParallelPolicy parallel_policy_;
          TargetQubit target_qubit_;
          ControlQubit control_qubit_;

          call_controlled_not(
            ParallelPolicy const parallel_policy,
            TargetQubit const target_qubit, ControlQubit const control_qubit)
            : parallel_policy_(parallel_policy),
              target_qubit_(target_qubit), control_qubit_(control_qubit)
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first,
            RandomAccessIterator const last) const
          {
            ::ket::gate::controlled_not(
              parallel_policy_, first, last, target_qubit_, control_qubit_);
          }
        };

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_controlled_not<ParallelPolicy, TargetQubit, ControlQubit>
        make_call_controlled_not(
          ParallelPolicy const parallel_policy,
          TargetQubit const target_qubit, ControlQubit const control_qubit)
        {
          return call_controlled_not<ParallelPolicy, TargetQubit, ControlQubit>(
            parallel_policy, target_qubit, control_qubit);
        }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      } // namespace controlled_not_detail

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_not(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("CNOT ", std::ios_base::ate);
        output_string_stream << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        KET_array<qubit_type, 2u> qubits = { target_qubit, control_qubit.qubit() };
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation,
          buffer, datatype, communicator, environment);

        if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::controlled_not_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit, permutation);

          return ::ket::mpi::gate::page::controlled_not_tp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation);
        }
        else if (::ket::mpi::page::is_on_page(control_qubit.qubit(), local_state, permutation))
          return ::ket::mpi::gate::page::controlled_not_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        qubit_type const permutated_target_qubit = permutation[target_qubit];
        ::ket::control<qubit_type> const permutated_control_qubit
          = ::ket::make_control(permutation[control_qubit.qubit()]);
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          [parallel_policy, permutated_target_qubit, permutated_control_qubit](
            auto const first, auto const last)
          {
            ::ket::gate::controlled_not(
              parallel_policy, first, last,
              permutated_target_qubit, permutated_control_qubit);
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          ::ket::mpi::gate::controlled_not_detail::make_call_controlled_not(
            parallel_policy,
            permutation[target_qubit],
            ::ket::make_control(permutation[control_qubit.qubit()])));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_not(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_not(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_not(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_not(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      namespace controlled_not_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_adj_controlled_not
        {
          ParallelPolicy parallel_policy_;
          TargetQubit target_qubit_;
          ControlQubit control_qubit_;

          call_adj_controlled_not(
            ParallelPolicy const parallel_policy,
            TargetQubit const target_qubit, ControlQubit const control_qubit)
            : parallel_policy_(parallel_policy),
              target_qubit_(target_qubit), control_qubit_(control_qubit)
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first,
            RandomAccessIterator const last) const
          {
            ::ket::gate::adj_controlled_not(
              parallel_policy_, first, last, target_qubit_, control_qubit_);
          }
        };

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_adj_controlled_not<ParallelPolicy, TargetQubit, ControlQubit>
        make_call_adj_controlled_not(
          ParallelPolicy const parallel_policy,
          TargetQubit const target_qubit, ControlQubit const control_qubit)
        {
          return call_adj_controlled_not<ParallelPolicy, TargetQubit, ControlQubit>(
            parallel_policy, target_qubit, control_qubit);
        }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_not(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Adj(CNOT) ", std::ios_base::ate);
        output_string_stream << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        KET_array<qubit_type, 2u> qubits = { target_qubit, control_qubit.qubit() };
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation,
          buffer, datatype, communicator, environment);

        if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::adj_controlled_not_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit, permutation);

          return ::ket::mpi::gate::page::adj_controlled_not_tp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation);
        }
        else if (::ket::mpi::page::is_on_page(control_qubit.qubit(), local_state, permutation))
          return ::ket::mpi::gate::page::adj_controlled_not_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit, permutation);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        qubit_type const permutated_target_qubit = permutation[target_qubit];
        ::ket::control<qubit_type> const permutated_control_qubit
          = ::ket::make_control(permutation[control_qubit.qubit()]);
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          [parallel_policy, permutated_target_qubit, permutated_control_qubit](
            auto const first, auto const last)
          {
            ::ket::gate::adj_controlled_not(
              parallel_policy, first, last,
              permutated_target_qubit, permutated_control_qubit);
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          ::ket::mpi::gate::controlled_not_detail::make_call_adj_controlled_not(
            parallel_policy,
            permutation[target_qubit],
            ::ket::make_control(permutation[control_qubit.qubit()])));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_not(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_not(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_not(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_not(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


# undef KET_array

#endif

