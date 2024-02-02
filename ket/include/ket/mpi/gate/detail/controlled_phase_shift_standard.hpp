#ifndef KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_STANDARD_HPP
# define KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_STANDARD_HPP

# include <boost/config.hpp>

# include <complex>
# include <vector>
# include <array>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/controlled_phase_shift_standard.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      // controlled_phase_shift_coeff
      // U_{tc}(s)
      // U_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{is} a_{11} |11>
      namespace controlled_phase_shift_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <
          typename ParallelPolicy, typename Complex,
          typename TargetQubit, typename ControlQubit>
        struct call_controlled_phase_shift_coeff
        {
          ParallelPolicy parallel_policy_;
          Complex phase_coefficient_;
          ::ket::mpi::permutated<TargetQubit> permutated_target_qubit_;
          ::ket::mpi::permutated<ControlQubit> permutated_control_qubit_;

          call_controlled_phase_shift_coeff(
            ParallelPolicy const parallel_policy, Complex const& phase_coefficient,
            ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
            ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
            : parallel_policy_{parallel_policy},
              phase_coefficient_{phase_coefficient},
              permutated_target_qubit_{permutated_target_qubit},
              permutated_control_qubit_{permutated_control_qubit}
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first, RandomAccessIterator const last) const
          {
            ::ket::gate::controlled_phase_shift_coeff(
              parallel_policy_, first, last, phase_coefficient_,
              permutated_target_qubit_.qubit(), permutated_control_qubit_.qubit());
          }
        }; // struct call_controlled_phase_shift_coeff<ParallelPolicy, Complex, TargetQubit, ControlQubit>

        template <
          typename ParallelPolicy, typename Complex,
          typename TargetQubit, typename ControlQubit>
        inline call_controlled_phase_shift_coeff<ParallelPolicy, Complex, TargetQubit, ControlQubit>
        make_call_controlled_phase_shift_coeff(
          ParallelPolicy const parallel_policy, Complex const& phase_coefficient,
          ::ket::mpi::permutated<TargetQubit> const permutated_target_qubit,
          ::ket::mpi::permutated<ControlQubit> const permutated_control_qubit)
        { return {parallel_policy, phase_coefficient, permutated_target_qubit, permutated_control_qubit}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& do_controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
              return ::ket::mpi::gate::page::controlled_phase_shift_coeff_tcp(
                parallel_policy, local_state, phase_coefficient, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::controlled_phase_shift_coeff_tp(
              parallel_policy, local_state, phase_coefficient, permutated_target_qubit, permutated_control_qubit);
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::controlled_phase_shift_coeff_cp(
              parallel_policy, local_state, phase_coefficient, permutated_target_qubit, permutated_control_qubit);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, &phase_coefficient, permutated_target_qubit, permutated_control_qubit](
              auto const first, auto const last)
            {
              ::ket::gate::controlled_phase_shift_coeff(
                parallel_policy, first, last, phase_coefficient,
                permutated_target_qubit.qubit(), permutated_control_qubit.qubit());
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            ::ket::mpi::gate::controlled_phase_shift_detail::make_call_controlled_phase_shift_coeff(
              parallel_policy, phase_coefficient, permutated_target_qubit, permutated_control_qubit));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 2u>{target_qubit, control_qubit.qubit()};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::controlled_phase_shift_detail::do_controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, target_qubit, control_qubit, permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubits = std::array<qubit_type, 2u>{target_qubit, control_qubit.qubit()};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::controlled_phase_shift_detail::do_controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, target_qubit, control_qubit, permutation, communicator, environment);
        }
      } // namespace controlled_phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Cphase(coeff) "}, phase_coefficient, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Cphase(coeff) "}, phase_coefficient, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace controlled_phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using std::conj;
          return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, conj(phase_coefficient),
            target_qubit, control_qubit, permutation, buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using std::conj;
          return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, conj(phase_coefficient),
            target_qubit, control_qubit, permutation, buffer, datatype, communicator, environment);
        }
      } // namespace controlled_phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Cphase(coeff)) "}, phase_coefficient, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::adj_controlled_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Cphase(coeff)) "}, phase_coefficient, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::adj_controlled_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      // controlled_phase_shift
      namespace controlled_phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase),
            target_qubit, control_qubit, permutation, buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase),
            target_qubit, control_qubit, permutation, buffer, datatype, communicator, environment);
        }
      } // namespace controlled_phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Cphase "}, phase, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Cphase "}, phase, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace controlled_phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift(
            mpi_policy, parallel_policy,
            local_state, -phase, target_qubit, control_qubit, permutation,
            buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::controlled_phase_shift_detail::controlled_phase_shift(
            mpi_policy, parallel_policy,
            local_state, -phase, target_qubit, control_qubit, permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace controlled_phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Cphase) "}, phase, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::adj_controlled_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(Cphase) "}, phase, ' ', target_qubit, ' ', control_qubit), environment};

        return ::ket::mpi::gate::controlled_phase_shift_detail::adj_controlled_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_STANDARD_HPP
