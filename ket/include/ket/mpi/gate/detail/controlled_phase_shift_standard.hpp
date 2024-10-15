#ifndef KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_STANDARD_HPP
# define KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_STANDARD_HPP

# include <complex>
# include <array>

# include <ket/gate/controlled_phase_shift.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/controlled_phase_shift_standard.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      // controlled_phase_shift_coeff
      // U_{tc}(theta)
      // U_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
      //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + e^{i theta} a_{11} |11>
      namespace local
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        [[deprecated]] inline auto controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        -> RandomAccessRange&
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment, permutated_target_qubit, permutated_control_qubit);

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

          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, communicator, environment,
            [parallel_policy, &phase_coefficient, permutated_target_qubit, permutated_control_qubit](
              auto const first, auto const last)
            {
              ::ket::gate::controlled_phase_shift_coeff(
                parallel_policy, first, last, phase_coefficient,
                permutated_target_qubit.qubit(), permutated_control_qubit.qubit());
            });
        }
      } // namespace local

      namespace controlled_phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        [[deprecated]] inline auto controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, target_qubit, control_qubit);

          return ::ket::mpi::gate::local::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, target_qubit, control_qubit, permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        [[deprecated]] inline auto controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, target_qubit, control_qubit);

          return ::ket::mpi::gate::local::controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, target_qubit, control_qubit, permutation, communicator, environment);
        }
      } // namespace controlled_phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      [[deprecated]] inline auto controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
        [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
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
        [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
        [[deprecated]] inline auto controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
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
        [[deprecated]] inline auto controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
        {
          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
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
      [[deprecated]] inline auto controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
        [[deprecated]] inline auto adj_controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
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
        [[deprecated]] inline auto adj_controlled_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const& phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator,
          yampi::environment const& environment)
        -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
      [[deprecated]] inline auto adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      -> RandomAccessRange&
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
