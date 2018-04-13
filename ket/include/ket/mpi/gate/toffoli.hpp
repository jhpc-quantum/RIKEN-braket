#ifndef KET_MPI_GATE_TOFFOLI_HPP
# define KET_MPI_GATE_TOFFOLI_HPP

# include <boost/config.hpp>

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif

# include <ket/qubit.hpp>
# include <ket/control.hpp>

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
      } // namespace toffoli_detail

      template <
        typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& toffoli(
        MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        KET_array<
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> >, 2u> const& control_qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype const datatype,
        yampi::communicator const communicator,
        yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print("Toffoli", environment);

        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
        KET_array<qubit_type, 3u> const qubits
          = { target_qubit, control_qubits[0u].qubit(), control_qubits[1u].qubit() };
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, qubits, permutation,
          buffer, datatype, communicator, environment);

        if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubits[0u].qubit(), local_state, permutation))
          {
            if (::ket::mpi::page::is_on_page(control_qubits[1u].qubit(), local_state, permutation))
              return ::ket::mpi::gate::page::controlled_
          }
        }
        else if
      }
    } // namespace gate

    template <
      typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange,
      typename StateInteger, typename BitInteger,
      typename Allocator, typename BufferAllocator>
    inline RandomAccessRange& toffoli(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      KET_array<
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> >, 2u> const& control_qubits,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print("Toffoli", environment);

      typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
      KET_array<qubit_type, 3u> const qubits
        = { target_qubit, control_qubits[0u].qubit(), control_qubits[1u].qubit() };
      ::ket::mpi::utility::maybe_interchange_qubits(
        mpi_policy, parallel_policy,
        local_state, qubits, permutation,
        buffer, datatype, communicator, environment);

      // TODO: page version
      if (::ket::mpi::page::is_on_page(target_qubit, local_state, permutation))
      {
        if (::ket::mpi::page::is_on_page(control_qubits[0u].qubit(), local_state, permutation))
        {
          if (::ket::mpi::page::is_on_page(control_qubits[1u].qubit(), local_state, permutation))
            return ::ket::mpi::gate::page::toffoli_tc0c1p(
              mpi_policy, parallel_policy, local_state,
              target_qubit, control_qubits, permutation);

          return ::ket::mpi::gate::page::toffoli_tc0p(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubits, permutation);
        }
        else if (::ket::mpi::page::is_on_page(control_qubits[1u].qubit(), local_state, permutation))
          return ::ket::mpi::gate::page::toffoli_tc1p(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubits, permutation);

        return ::ket::mpi::gate::page::toffoli_tp(
          mpi_policy, parallel_policy, local_state,
          target_qubit, control_qubits, permutation);
      }
      else if (::ket::mpi::page::is_on_page(control_qubits[0u].qubit(), local_state, permutation))
      {
        if (::ket::mpi::page::is_on_page(control_qubits[1u].qubit(), local_state, permutation))
          return ::ket::mpi::gate::page::toffoli_c0c1p(
            mpi_policy, parallel_policy, local_state,
            target_qubit, control_qubits, permutation);

        return ::ket::mpi::gate::page::toffoli_c0p(
          mpi_policy, parallel_policy, local_state,
          target_qubit, control_qubits, permutation);
      }
      else if (::ket::mpi::page::is_on_page(control_qubits[1u].qubit(), local_state, permutation))
        return ::ket::mpi::gate::page::toffoli_c1p(
          mpi_policy, parallel_policy, local_state,
          target_qubit, control_qubits, permutation);

      KET_array< ::ket:control<qubit_type>, 2u> const permutated_control_qubits
        = { ::ket::make_control(permutation[control_qubits[0u]]),
            ::ket::make_control(permutation[control_qubits[1u]]) };
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
      return ::ket::mpi::utility::for_each_local_range(
        mpi_policy, local_state,
        [parallel_policy, target_qubit, &permutated_control_qubits, &permutation](
          auto const first, auto const last)
        {
          ::ket::gate::toffoli(
            parallel_policy, first, last,
            permutation[target_qubit], permutated_control_qubits);
        });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
      return ::ket::mpi::utility::for_each_local_range(
        mpi_policy, local_state,
        ::ket::mpi::gate::toffoli_detail::make_call_toffoli(
          parallel_policy, permutation[target_qubit], permutated_control_qubits));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
    }
  } // namespace mpi
} // namespace ket


# undef KET_array

#endif
