#ifndef KET_MPI_GATE_CONTROLLED_V_HPP
# define KET_MPI_GATE_CONTROLLED_V_HPP

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
# include <ket/gate/controlled_v.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/controlled_v.hpp>
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
      // controlled_v_coeff
      namespace controlled_v_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <
          typename ParallelPolicy, typename Complex,
          typename TargetQubit, typename ControlQubit>
        struct call_controlled_v_coeff
        {
          ParallelPolicy parallel_policy_;
          Complex phase_coefficient_;
          TargetQubit target_qubit_;
          ControlQubit control_qubit_;

          call_controlled_v_coeff(
            ParallelPolicy const parallel_policy, Complex const& phase_coefficient,
            TargetQubit const target_qubit, ControlQubit const control_qubit)
            : parallel_policy_(parallel_policy), phase_coefficient_(phase_coefficient),
              target_qubit_(target_qubit), control_qubit_(control_qubit)
          { }

          template <typename RandomAccessIterator>
          void operator()(
            RandomAccessIterator const first,
            RandomAccessIterator const last) const
          {
            ::ket::gate::controlled_v_coeff(
              parallel_policy_, first, last, phase_coefficient_, target_qubit_, control_qubit_);
          }
        };

        template <
          typename ParallelPolicy, typename Complex,
          typename TargetQubit, typename ControlQubit>
        inline call_controlled_v_coeff<ParallelPolicy, Complex, TargetQubit, ControlQubit>
        make_call_controlled_v_coeff(
          ParallelPolicy const parallel_policy, Complex const& phase_coefficient,
          TargetQubit const target_qubit, ControlQubit const control_qubit)
        {
          typedef
            call_controlled_v_coeff<ParallelPolicy, Complex, TargetQubit, ControlQubit>
            result_type;
          return result_type(parallel_policy, phase_coefficient, target_qubit, control_qubit);
        }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& controlled_v_coeff(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
          KET_array<qubit_type, 2u> qubits = { target_qubit, control_qubit.qubit() };
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubits, permutation,
            buffer, datatype, communicator, environment);

          if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
          {
            if (::ket::mpi::page::is_on_page(control_qubit.qubit(), local_state, permutation))
              return ::ket::mpi::gate::page::controlled_v_coeff_tcp(
                mpi_policy, parallel_policy,
                local_state, phase_coefficient, target_qubit, control_qubit, permutation);

            return ::ket::mpi::gate::page::controlled_v_coeff_tp(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient, target_qubit, control_qubit, permutation);
          }
          else if (::ket::mpi::page::is_on_page(
                     control_qubit.qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::controlled_v_coeff_cp(
              mpi_policy, parallel_policy,
              local_state, phase_coefficient, target_qubit, control_qubit, permutation);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state,
            [parallel_policy, &phase_coefficient, target_qubit, control_qubit, &permutation](
              auto const first, auto const last)
            {
              ::ket::gate::controlled_v_coeff(
                parallel_policy, first, last, phase_coefficient,
                permutation[target_qubit],
                ::ket::make_control(permutation[control_qubit.qubit()]));
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state,
            ::ket::mpi::gate::controlled_v_detail::make_call_controlled_v_coeff(
              parallel_policy, phase_coefficient,
              permutation[target_qubit],
              ::ket::make_control(permutation[control_qubit.qubit()])));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace controlled_v_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_v_coeff(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Cv(coeff) ", std::ios_base::ate);
        output_string_stream << phase_coefficient << ' ' << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);
        return ::ket::mpi::gate::controlled_v_detail::controlled_v_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_v_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_v_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_v_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      namespace controlled_v_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& conj_controlled_v_coeff(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::controlled_v_detail::controlled_v_coeff(
            mpi_policy, parallel_policy,
            local_state, static_cast<Complex>(1)/phase_coefficient,
            target_qubit, control_qubit, permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace controlled_v_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_controlled_v_coeff(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Conj(Cv(coeff)) ", std::ios_base::ate);
        output_string_stream << phase_coefficient << ' ' << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);
        return ::ket::mpi::gate::controlled_v_detail::conj_controlled_v_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_controlled_v_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_controlled_v_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_controlled_v_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_v_coeff(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Adj(Cv(coeff)) ", std::ios_base::ate);
        output_string_stream << phase_coefficient << ' ' << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);
        return ::ket::mpi::gate::controlled_v_detail::conj_controlled_v_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_v_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_v_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_v_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_v_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase_coefficient, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      // controlled_v
      namespace controlled_v_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& controlled_v(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          typedef typename boost::range_value<RandomAccessRange>::type complex_type;
          return ::ket::mpi::gate::controlled_v_detail::controlled_v_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase),
            target_qubit, control_qubit, permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace controlled_v_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_v(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Cv ", std::ios_base::ate);
        output_string_stream << phase << ' ' << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);
        return ::ket::mpi::gate::controlled_v_detail::controlled_v(
          mpi_policy, parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_v(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_v(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::controlled_v(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      namespace controlled_v_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& conj_controlled_v(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::controlled_v_detail::controlled_v(
            mpi_policy, parallel_policy,
            local_state, -phase, target_qubit, control_qubit, permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace controlled_v_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_controlled_v(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Conj(Cv) ", std::ios_base::ate);
        output_string_stream << phase << ' ' << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);
        return ::ket::mpi::gate::controlled_v_detail::conj_controlled_v(
          mpi_policy, parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_controlled_v(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_controlled_v(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_controlled_v(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_v(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        std::ostringstream output_string_stream("Adj(Cv) ", std::ios_base::ate);
        output_string_stream << phase << ' ' << target_qubit << ' ' << control_qubit;
        ::ket::mpi::utility::log_with_time_guard<char> print(output_string_stream.str(), environment);
        return ::ket::mpi::gate::controlled_v_detail::conj_controlled_v(
          mpi_policy, parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_v(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_v(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_controlled_v(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_controlled_v(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase, target_qubit, control_qubit, permutation,
          buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


# undef KET_array

#endif

