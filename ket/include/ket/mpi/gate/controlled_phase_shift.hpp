#ifndef KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_HPP
# define KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <vector>

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
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/phase_shift.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>


#ifndef KET_USE_DIAGONAL_LOOP
# include <ket/mpi/gate/detail/controlled_phase_shift_standard.hpp>
#else // KET_USE_DIAGONAL_LOOP
# include <ket/mpi/gate/detail/controlled_phase_shift_diagonal.hpp>
#endif // KET_USE_DIAGONAL_LOOP


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
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string(sizeof...(ControlQubits) + 1u, 'C').append("phase(coeff) "), phase_coefficient),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string(sizeof...(ControlQubits) + 1u, 'C').append("phase(coeff) "), phase_coefficient),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string{"Adj("}.append(sizeof...(ControlQubits) + 1u, 'C').append("phase(coeff)) "), phase_coefficient),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string{"Adj("}.append(sizeof...(ControlQubits) + 1u, 'C').append("phase(coeff)) "), phase_coefficient),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift_coeff(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, phase_coefficient,
          target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Complex const& phase_coefficient,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase_coefficient, target_qubit, control_qubit, control_qubits...);
      }

      // controlled_phase_shift
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Real, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string(sizeof...(ControlQubits) + 1u, 'C').append("phase "), phase),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string(sizeof...(ControlQubits) + 1u, 'C').append("phase "), phase),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::phase_shift(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Real, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Real, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename... ControlQubits>
      inline RandomAccessRange& controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Real, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string{"Adj("}.append(sizeof...(ControlQubits) + 1u, 'C').append("phase) "), phase),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(
            ::ket::mpi::utility::generate_logger_string(
              std::string{"Adj("}.append(sizeof...(ControlQubits) + 1u, 'C').append("phase) "), phase),
            target_qubit, control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::phase_shift_detail::adj_phase_shift(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Real, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename Real, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename... ControlQubits>
      inline RandomAccessRange& adj_controlled_phase_shift(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ControlQubits const... control_qubits)
      {
        return ::ket::mpi::gate::adj_controlled_phase_shift(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment,
          phase, target_qubit, control_qubit, control_qubits...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket



#endif // KET_MPI_GATE_DETAIL_CONTROLLED_PHASE_SHIFT_HPP
