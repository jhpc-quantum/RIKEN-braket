#ifndef KET_MPI_GATE_TOFFOLI_HPP
# define KET_MPI_GATE_TOFFOLI_HPP

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
# include <ket/gate/toffoli.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/toffoli.hpp>
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
      namespace toffoli_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_toffoli
        {
          ParallelPolicy parallel_policy_;
          TargetQubit permutated_target_qubit_;
          ControlQubit permutated_control_qubit1_;
          ControlQubit permutated_control_qubit2_;

          call_toffoli(
            ParallelPolicy const parallel_policy,
            TargetQubit const permutated_target_qubit,
            ControlQubit const permutated_control_qubit1,
            ControlQubit const permutated_control_qubit2)
            : parallel_policy_(parallel_policy),
              permutated_target_qubit_(permutated_target_qubit),
              permutated_control_qubit1_(permutated_control_qubit1),
              permutated_control_qubit2_(permutated_control_qubit2)
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first,
            RandomAccessIterator const last) const
          {
            ::ket::gate::toffoli(
              parallel_policy_, first, last,
              permutated_target_qubit_, permutated_control_qubit1_, permutated_control_qubit2_);
          }
        }; // struct call_toffoli<ParallelPolicy, TargetQubit, ControlQubit>

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_toffoli<ParallelPolicy, TargetQubit, ControlQubit>
        make_call_toffoli(
          ParallelPolicy const parallel_policy,
          TargetQubit const permutated_target_qubit,
          ControlQubit const permutated_control_qubit1, ControlQubit const permutated_control_qubit2)
        {
          return call_toffoli<ParallelPolicy, TargetQubit, ControlQubit>(
            parallel_policy,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
        }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      } // namespace toffoli_detail

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& toffoli(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Toffoli ", std::ios_base::ate);
        output_string_stream << target_qubit << ' ' << control_qubit1 << ' ' << control_qubit2;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        KET_array<qubit_type, 3u> const qubits
          = { target_qubit, control_qubit1.qubit(), control_qubit2.qubit() };
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation,
          buffer, datatype, communicator, environment);

        if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit1.qubit(), local_state, permutation))
          {
            if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
              return ::ket::mpi::gate::page::toffoli_tccp(
                mpi_policy, parallel_policy, local_state,
                target_qubit, control_qubit1, control_qubit2, permutation);

            return ::ket::mpi::gate::page::toffoli_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit1, control_qubit2, permutation);
          }

          if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::toffoli_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit2, control_qubit1, permutation);

          return ::ket::mpi::gate::page::toffoli_tp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }
        else if(::ket::mpi::page::is_on_page(control_qubit1.qubit(), local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::toffoli_ccp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit1, control_qubit2, permutation);

          return ::ket::mpi::gate::page::toffoli_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }
        else if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
          return ::ket::mpi::gate::page::toffoli_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit2, control_qubit1, permutation);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        qubit_type permutated_target_qubit = permutation[target_qubit];
        typedef ::ket::control<qubit_type> control_qubit_type;
        control_qubit_type permutated_control_qubit1
          = ::ket::make_control(permutation[control_qubit1.qubit()]);
        control_qubit_type permutated_control_qubit2
          = ::ket::make_control(permutation[control_qubit2.qubit()]);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          [parallel_policy, permutated_target_qubit,
           permutated_control_qubit1, permutated_control_qubit2](
            auto const first, auto const last)
          {
            ::ket::gate::toffoli(
              parallel_policy, first, last,
              permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          ::ket::mpi::gate::toffoli_detail::make_call_toffoli(
            parallel_policy,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& toffoli(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::toffoli(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, target_qubit, control_qubit1, control_qubit2, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& toffoli(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::toffoli(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, target_qubit, control_qubit1, control_qubit2, permutation,
          buffer, datatype, communicator, environment);
      }


      namespace toffoli_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_conj_toffoli
        {
          ParallelPolicy parallel_policy_;
          TargetQubit permutated_target_qubit_;
          ControlQubit permutated_control_qubit1_;
          ControlQubit permutated_control_qubit2_;

          call_conj_toffoli(
            ParallelPolicy const parallel_policy,
            TargetQubit const permutated_target_qubit,
            ControlQubit const permutated_control_qubit1,
            ControlQubit const permutated_control_qubit2)
            : parallel_policy_(parallel_policy),
              permutated_target_qubit_(permutated_target_qubit),
              permutated_control_qubit1_(permutated_control_qubit1),
              permutated_control_qubit2_(permutated_control_qubit2)
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first,
            RandomAccessIterator const last) const
          {
            ::ket::gate::conj_toffoli(
              parallel_policy_, first, last,
              permutated_target_qubit_, permutated_control_qubit1_, permutated_control_qubit2_);
          }
        }; // struct call_conj_toffoli<ParallelPolicy, TargetQubit, ControlQubit>

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_conj_toffoli<ParallelPolicy, TargetQubit, ControlQubit>
        make_call_conj_toffoli(
          ParallelPolicy const parallel_policy,
          TargetQubit const permutated_target_qubit,
          ControlQubit const permutated_control_qubit1, ControlQubit const permutated_control_qubit2)
        {
          return call_conj_toffoli<ParallelPolicy, TargetQubit, ControlQubit>(
            parallel_policy,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
        }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      } // namespace toffoli_detail

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_toffoli(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Conj(Toffoli) ", std::ios_base::ate);
        output_string_stream << target_qubit << ' ' << control_qubit1 << ' ' << control_qubit2;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        KET_array<qubit_type, 3u> const qubits
          = { target_qubit, control_qubit1.qubit(), control_qubit2.qubit() };
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation,
          buffer, datatype, communicator, environment);

        if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit1.qubit(), local_state, permutation))
          {
            if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
              return ::ket::mpi::gate::page::conj_toffoli_tccp(
                mpi_policy, parallel_policy, local_state,
                target_qubit, control_qubit1, control_qubit2, permutation);

            return ::ket::mpi::gate::page::conj_toffoli_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit1, control_qubit2, permutation);
          }

          if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::conj_toffoli_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit2, control_qubit1, permutation);

          return ::ket::mpi::gate::page::conj_toffoli_tp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }
        else if(::ket::mpi::page::is_on_page(control_qubit1.qubit(), local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::conj_toffoli_ccp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit1, control_qubit2, permutation);

          return ::ket::mpi::gate::page::conj_toffoli_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }
        else if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
          return ::ket::mpi::gate::page::conj_toffoli_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit2, control_qubit1, permutation);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        qubit_type permutated_target_qubit = permutation[target_qubit];
        typedef ::ket::control<qubit_type> control_qubit_type;
        control_qubit_type permutated_control_qubit1
          = ::ket::make_control(permutation[control_qubit1.qubit()]);
        control_qubit_type permutated_control_qubit2
          = ::ket::make_control(permutation[control_qubit2.qubit()]);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          [parallel_policy, permutated_target_qubit,
           permutated_control_qubit1, permutated_control_qubit2](
            auto const first, auto const last)
          {
            ::ket::gate::conj_toffoli(
              parallel_policy, first, last,
              permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          ::ket::mpi::gate::toffoli_detail::make_call_conj_toffoli(
            parallel_policy,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_toffoli(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_toffoli(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, target_qubit, control_qubit1, control_qubit2, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_toffoli(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_toffoli(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, target_qubit, control_qubit1, control_qubit2, permutation,
          buffer, datatype, communicator, environment);
      }


      namespace toffoli_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        struct call_adj_toffoli
        {
          ParallelPolicy parallel_policy_;
          TargetQubit permutated_target_qubit_;
          ControlQubit permutated_control_qubit1_;
          ControlQubit permutated_control_qubit2_;

          call_adj_toffoli(
            ParallelPolicy const parallel_policy,
            TargetQubit const permutated_target_qubit,
            ControlQubit const permutated_control_qubit1,
            ControlQubit const permutated_control_qubit2)
            : parallel_policy_(parallel_policy),
              permutated_target_qubit_(permutated_target_qubit),
              permutated_control_qubit1_(permutated_control_qubit1),
              permutated_control_qubit2_(permutated_control_qubit2)
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first,
            RandomAccessIterator const last) const
          {
            ::ket::gate::adj_toffoli(
              parallel_policy_, first, last,
              permutated_target_qubit_, permutated_control_qubit1_, permutated_control_qubit2_);
          }
        }; // struct call_adj_toffoli<ParallelPolicy, TargetQubit, ControlQubit>

        template <typename ParallelPolicy, typename TargetQubit, typename ControlQubit>
        inline call_adj_toffoli<ParallelPolicy, TargetQubit, ControlQubit>
        make_call_adj_toffoli(
          ParallelPolicy const parallel_policy,
          TargetQubit const permutated_target_qubit,
          ControlQubit const permutated_control_qubit1, ControlQubit const permutated_control_qubit2)
        {
          return call_adj_toffoli<ParallelPolicy, TargetQubit, ControlQubit>(
            parallel_policy,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
        }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      } // namespace toffoli_detail

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_toffoli(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Adj(Toffoli) ", std::ios_base::ate);
        output_string_stream << target_qubit << ' ' << control_qubit1 << ' ' << control_qubit2;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        KET_array<qubit_type, 3u> const qubits
          = { target_qubit, control_qubit1.qubit(), control_qubit2.qubit() };
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation,
          buffer, datatype, communicator, environment);

        if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit1.qubit(), local_state, permutation))
          {
            if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
              return ::ket::mpi::gate::page::adj_toffoli_tccp(
                mpi_policy, parallel_policy, local_state,
                target_qubit, control_qubit1, control_qubit2, permutation);

            return ::ket::mpi::gate::page::adj_toffoli_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit1, control_qubit2, permutation);
          }

          if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::adj_toffoli_tcp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit2, control_qubit1, permutation);

          return ::ket::mpi::gate::page::adj_toffoli_tp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }
        else if(::ket::mpi::page::is_on_page(control_qubit1.qubit(), local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::adj_toffoli_ccp(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubit1, control_qubit2, permutation);

          return ::ket::mpi::gate::page::adj_toffoli_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit1, control_qubit2, permutation);
        }
        else if (::ket::mpi::page::is_on_page(control_qubit2.qubit(), local_state, permutation))
          return ::ket::mpi::gate::page::adj_toffoli_cp(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubit2, control_qubit1, permutation);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        qubit_type permutated_target_qubit = permutation[target_qubit];
        typedef ::ket::control<qubit_type> control_qubit_type;
        control_qubit_type permutated_control_qubit1
          = ::ket::make_control(permutation[control_qubit1.qubit()]);
        control_qubit_type permutated_control_qubit2
          = ::ket::make_control(permutation[control_qubit2.qubit()]);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          [parallel_policy, permutated_target_qubit,
           permutated_control_qubit1, permutated_control_qubit2](
            auto const first, auto const last)
          {
            ::ket::gate::adj_toffoli(
              parallel_policy, first, last,
              permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2);
          });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
        return ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state,
          ::ket::mpi::gate::toffoli_detail::make_call_adj_toffoli(
            parallel_policy,
            permutated_target_qubit, permutated_control_qubit1, permutated_control_qubit2));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
      }

      template <
        typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_toffoli(
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_toffoli(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, target_qubit, control_qubit1, control_qubit2, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_toffoli(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_toffoli(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, target_qubit, control_qubit1, control_qubit2, permutation,
          buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


# undef KET_array

#endif
