#ifndef KET_MPI_GATE_DETAIL_PHASE_SHIFT_STANDARD_HPP
# define KET_MPI_GATE_DETAIL_PHASE_SHIFT_STANDARD_HPP

# include <boost/config.hpp>

# include <complex>
# include <vector>
# include <array>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/phase_shift.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/phase_shift.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      // phase_shift_coeff
      namespace phase_shift_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename Complex, typename Qubit>
        struct call_phase_shift_coeff
        {
          ParallelPolicy parallel_policy_;
          Complex phase_coefficient_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_phase_shift_coeff(
            ParallelPolicy const parallel_policy,
            Complex const& phase_coefficient, ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy},
              phase_coefficient_{phase_coefficient},
              permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first, RandomAccessIterator const last) const
          {
            ::ket::gate::phase_shift_coeff(
              parallel_policy_, first, last, phase_coefficient_, permutated_qubit_.qubit());
          }
        }; // struct call_phase_shift_coeff<ParallelPolicy, Complex, Qubit>

        template <typename ParallelPolicy, typename Complex, typename Qubit>
        inline call_phase_shift_coeff<ParallelPolicy, Complex, Qubit>
        make_call_phase_shift_coeff(
          ParallelPolicy const parallel_policy,
          Complex const& phase_coefficient, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, phase_coefficient, permutated_qubit}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& do_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::phase_shift_coeff(
              parallel_policy, local_state, phase_coefficient, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, &phase_coefficient, permutated_qubit](
              auto const first, auto const last)
            {
              ::ket::gate::phase_shift_coeff(
                parallel_policy, first, last, phase_coefficient, permutated_qubit.qubit());
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::phase_shift_detail::make_call_phase_shift_coeff(
              parallel_policy, phase_coefficient, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, phase_coefficient, qubit, permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, phase_coefficient, qubit, permutation, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase(coeff) "}, phase_coefficient, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase(coeff) "}, phase_coefficient, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using std::conj;
          return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, conj(phase_coefficient), qubit, permutation, buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using std::conj;
          return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, conj(phase_coefficient), qubit, permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase(coeff)) "}, phase_coefficient, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase(coeff)) "}, phase_coefficient, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      // phase_shift
      namespace phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase), qubit, permutation,
            buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase), qubit, permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      namespace phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::phase_shift_detail::phase_shift(
            mpi_policy, parallel_policy,
            local_state, -phase, qubit, permutation, buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::phase_shift_detail::phase_shift(
            mpi_policy, parallel_policy,
            local_state, -phase, qubit, permutation, buffer, datatype, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      // generalized phase_shift
      namespace phase_shift_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename Real, typename Qubit>
        struct call_phase_shift2
        {
          ParallelPolicy parallel_policy_;
          Real phase1_;
          Real phase2_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_phase_shift2(
            ParallelPolicy const parallel_policy,
            Real const phase1, Real const phase2, ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy},
              phase1_{phase1},
              phase2_{phase2},
              permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first, RandomAccessIterator const last) const
          {
            ::ket::gate::phase_shift2(
              parallel_policy_, first, last, phase1_, phase2_, permutated_qubit_.qubit());
          }
        }; // struct call_phase_shift2<ParallelPolicy, Real, Qubit>

        template <typename ParallelPolicy, typename Real, typename Qubit>
        inline call_phase_shift2<ParallelPolicy, Real, Qubit>
        make_call_phase_shift2(
          ParallelPolicy const parallel_policy,
          Real const phase1, Real const phase2, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, phase1, phase2, permutated_qubit}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& do_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::phase_shift2(
              parallel_policy, local_state, phase1, phase2, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, phase1, phase2, permutated_qubit](
              auto const first, auto const last)
            {
              ::ket::gate::phase_shift2(
                parallel_policy, first, last, phase1, phase2, permutated_qubit.qubit());
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::phase_shift_detail::make_call_phase_shift2(
              parallel_policy, phase1, phase2, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_phase_shift2(
            mpi_policy, parallel_policy, local_state, phase1, phase2, qubit, permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_phase_shift2(
            mpi_policy, parallel_policy, local_state, phase1, phase2, qubit, permutation, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift2(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase1, ' ', phase2, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift2(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift2(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase1, ' ', phase2, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift2(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift2(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift2(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace phase_shift_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename Real, typename Qubit>
        struct call_adj_phase_shift2
        {
          ParallelPolicy parallel_policy_;
          Real phase1_;
          Real phase2_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_adj_phase_shift2(
            ParallelPolicy const parallel_policy,
            Real const phase1, Real const phase2, ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy},
              phase1_{phase1},
              phase2_{phase2},
              permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first, RandomAccessIterator const last) const
          {
            ::ket::gate::adj_phase_shift2(
              parallel_policy_, first, last, phase1_, phase2_, permutated_qubit_.qubit());
          }
        }; // struct call_adj_phase_shift2

        template <typename ParallelPolicy, typename Real, typename Qubit>
        inline call_adj_phase_shift2<ParallelPolicy, Real, Qubit>
        make_call_adj_phase_shift2(
          ParallelPolicy const parallel_policy,
          Real const phase1, Real const phase2, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, phase1, phase2, permutated_qubit}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& do_adj_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::adj_phase_shift2(
              parallel_policy, local_state, phase1, phase2, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, phase1, phase2, permutated_qubit](
              auto const first, auto const last)
            {
              ::ket::gate::adj_phase_shift2(
                parallel_policy, first, last, phase1, phase2, permutated_qubit.qubit());
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::phase_shift_detail::make_call_adj_phase_shift2(
              parallel_policy, phase1, phase2, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_adj_phase_shift2(
            mpi_policy, parallel_policy, local_state, phase1, phase2, qubit, permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_adj_phase_shift2(
            mpi_policy, parallel_policy, local_state, phase1, phase2, qubit, permutation, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift2(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase1, ' ', phase2, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift2(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift2(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase1, ' ', phase2, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift2(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift2(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift2(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift2(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift2(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, qubit, permutation, buffer, datatype, communicator, environment);
      }

      namespace phase_shift_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename Real, typename Qubit>
        struct call_phase_shift3
        {
          ParallelPolicy parallel_policy_;
          Real phase1_;
          Real phase2_;
          Real phase3_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_phase_shift3(
            ParallelPolicy const parallel_policy,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy},
              phase1_{phase1},
              phase2_{phase2},
              phase3_{phase3},
              permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first, RandomAccessIterator const last) const
          {
            ::ket::gate::phase_shift3(
              parallel_policy_, first, last, phase1_, phase2_, phase3_, permutated_qubit_.qubit());
          }
        }; // struct call_phase_shift3<ParallelPolicy, Real, Qubit>

        template <typename ParallelPolicy, typename Real, typename Qubit>
        inline call_phase_shift3<ParallelPolicy, Real, Qubit>
        make_call_phase_shift3(
          ParallelPolicy const parallel_policy,
          Real const phase1, Real const phase2, Real const phase3, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, phase1, phase2, phase3, permutated_qubit}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& do_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::phase_shift3(
              parallel_policy, local_state, phase1, phase2, phase3, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, phase1, phase2, phase3, permutated_qubit](
              auto const first, auto const last)
            {
              ::ket::gate::phase_shift3(
                parallel_policy, first, last, phase1, phase2, phase3, permutated_qubit.qubit());
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::phase_shift_detail::make_call_phase_shift3(
              parallel_policy, phase1, phase2, phase3, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_phase_shift3(
            mpi_policy, parallel_policy, local_state, phase1, phase2, phase3, qubit, permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_phase_shift3(
            mpi_policy, parallel_policy, local_state, phase1, phase2, phase3, qubit, permutation, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift3(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase1, ' ', phase2, ' ', phase3, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift3(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift3(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase1, ' ', phase2, ' ', phase3, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift3(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift3(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, phase3, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift3(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, phase3, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace phase_shift_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename ParallelPolicy, typename Real, typename Qubit>
        struct call_adj_phase_shift3
        {
          ParallelPolicy parallel_policy_;
          Real phase1_;
          Real phase2_;
          Real phase3_;
          ::ket::mpi::permutated<Qubit> permutated_qubit_;

          call_adj_phase_shift3(
            ParallelPolicy const parallel_policy,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::mpi::permutated<Qubit> const permutated_qubit)
            : parallel_policy_{parallel_policy},
              phase1_{phase1},
              phase2_{phase2},
              phase3_{phase3},
              permutated_qubit_{permutated_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first, RandomAccessIterator const last) const
          {
            ::ket::gate::adj_phase_shift3(
              parallel_policy_, first, last, phase1_, phase2_, phase3_, permutated_qubit_.qubit());
          }
        }; // struct call_adj_phase_shift3<ParallelPolicy, Real, Qubit>

        template <typename ParallelPolicy, typename Real, typename Qubit>
        inline call_adj_phase_shift3<ParallelPolicy, Real, Qubit>
        make_call_adj_phase_shift3(
          ParallelPolicy const parallel_policy,
          Real const phase1, Real const phase2, Real const phase3, ::ket::mpi::permutated<Qubit> const permutated_qubit)
        { return {parallel_policy, phase1, phase2, phase3, permutated_qubit}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& do_adj_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::adj_phase_shift3(
              parallel_policy, local_state, phase1, phase2, phase3, permutated_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, phase1, phase2, phase3, permutated_qubit](
              auto const first, auto const last)
            {
              ::ket::gate::adj_phase_shift3(
                parallel_policy, first, last, phase1, phase2, phase3, permutated_qubit.qubit());
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::phase_shift_detail::make_call_adj_phase_shift3(
              parallel_policy, phase1, phase2, phase3, permutated_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_adj_phase_shift3(
            mpi_policy, parallel_policy, local_state, phase1, phase2, phase3, qubit, permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase1, Real const phase2, Real const phase3,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 1u>{qubit};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::phase_shift_detail::do_adj_phase_shift3(
            mpi_policy, parallel_policy, local_state, phase1, phase2, phase3, qubit, permutation, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift3(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase1, ' ', phase2, ' ', phase3, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift3(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift3(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase1, ' ', phase2, ' ', phase3, ' ', qubit), environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift3(
          mpi_policy, parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift3(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, phase3, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift3(
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase1, phase2, phase3, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_phase_shift3(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase1, Real const phase2, Real const phase3,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift3(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase1, phase2, phase3, qubit, permutation,
          buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_PHASE_SHIFT_STANDARD_HPP
