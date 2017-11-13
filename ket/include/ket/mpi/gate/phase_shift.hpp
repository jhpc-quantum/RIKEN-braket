#ifndef KET_USE_DIAGONAL_LOOP
# include <ket/mpi/gate/phase_shift.hpp.old>
#else // KET_USE_DIAGONAL_LOOP
//
#ifndef KET_MPI_GATE_PHASE_SHIFT_HPP
# define KET_MPI_GATE_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <vector>

# include <boost/range/value_type.hpp>
# include <boost/range/iterator.hpp>
# include <boost/range/begin.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/gate/phase_shift.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
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
# ifdef BOOST_NO_CXX11_LAMBDAS
        struct just_return
        {
          template <typename StateInteger>
          void operator()(StateInteger const) const { }
        };

        template <typename Iterator, typename Complex>
        class do_phase_shift_coeff
        {
          Iterator local_state_first_;
          Complex phase_coefficient_;

         public:
          typedef void result_type;

          do_phase_shift_coeff(Iterator const local_state_first, Complex const& phase_coefficient)
            : local_state_first_(local_state_first), phase_coefficient_(phase_coefficient)
          { }

          template <typename StateInteger>
          void operator()(StateInteger const index) const
          { *(local_state_first_+index) *= phase_coefficient_; },
        };

        template <typename Iterator, typename Complex>
        inline
        do_phase_shift_coeff<Iterator, Complex> make_do_phase_shift_coeff(
          Iterator const local_state_first, Complex const& phase_coefficient)
        { return do_phase_shift_coeff<Iterator, Complex>(local_state_first, phase_coefficient); }
# endif // BOOST_NO_CXX11_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift_coeff(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          if (::ket::mpi::page::is_on_page(qubit, local_state, permutation))
            return ::ket::mpi::gate::page::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, phase_coefficient, qubit, permutation);

          typedef typename boost::range_iterator<RandomAccessRange>::type iterator;
          iterator local_state_first = boost::begin(local_state);

# ifndef BOOST_NO_CXX11_LAMBDAS
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            [](StateInteger const){},
            [local_state_first, &phase_coefficient](StateInteger const index)
            { *(local_state_first+index) *= phase_coefficient; });
# else // BOOST_NO_CXX11_LAMBDAS
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            ::ket::mpi::gate::phase_shift_detail::just_return(),
            ::ket::mpi::gate::phase_shift_detail::make_do_phase_shift_coeff(
              local_state_first, phase_coefficient));
# endif // BOOST_NO_CXX11_LAMBDAS

          return local_state;
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift_coeff(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print("Phase", environment);
        return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          datatype, communicator, environment);
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(),
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      namespace phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& conj_phase_shift_coeff(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, static_cast<Complex>(1)/phase_coefficient, qubit, permutation,
            datatype, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_phase_shift_coeff(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print("Conj(Phase)", environment);
        return ::ket::mpi::gate::phase_shift_detail::conj_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_phase_shift_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift_coeff(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print("Adj(Phase)", environment);
        return ::ket::mpi::gate::phase_shift_detail::conj_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          datatype, communicator, environment);
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(),
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      // phase_shift
      namespace phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          typedef typename boost::range_value<RandomAccessRange>::type complex_type;
          return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase), qubit, permutation,
            datatype, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& phase_shift(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>&,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print("Phase", environment);
        return ::ket::mpi::gate::phase_shift_detail::phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation,
          datatype, communicator, environment);
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, permutation,
          buffer, datatype, communicator, environment);
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::phase_shift(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase, qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      namespace phase_shift_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& conj_phase_shift(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::datatype const datatype,
          yampi::communicator const communicator,
          yampi::environment const& environment)
        {
          return ::ket::mpi::gate::phase_shift_detail::phase_shift(
            mpi_policy, parallel_policy,
            local_state, -phase, qubit, permutation,
            datatype, communicator, environment);
        }
      } // namespace phase_shift_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_phase_shift(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print("Conj(Phase)", environment);
        return ::ket::mpi::gate::phase_shift_detail::conj_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation,
          datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_phase_shift(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_phase_shift(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& conj_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::conj_phase_shift(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase, qubit, permutation,
          buffer, datatype, communicator, environment);
      }


      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_phase_shift(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print("Adj(Phase)", environment);
        return ::ket::mpi::gate::phase_shift_detail::conj_phase_shift(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation,
          datatype, communicator, environment);
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, permutation,
          buffer, datatype, communicator, environment);
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
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_phase_shift(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, phase, qubit, permutation,
          buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif
//
#endif // KET_USE_DIAGONAL_LOOP

