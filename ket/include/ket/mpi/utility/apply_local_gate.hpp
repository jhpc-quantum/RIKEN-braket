#ifndef KET_MPI_UTILITY_APPLY_LOCAL_GATE_HPP
# define KET_MPI_UTILITY_APPLY_LOCAL_GATE_HPP

# include <cstddef>
# include <vector>
# include <utility>

# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>

# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/environment.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace apply_local_gate_detail
      {
        template <std::size_t index, std::size_t num_operated_qubits>
        struct apply_local_gate
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator,
            typename LocalGate, typename... Qubits>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const global_qubit_value,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_unit_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_global_qubit,
            LocalGate&& local_gate, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits&&... qubits)
          -> RandomAccessRange&
          {
            return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              unit_control_qubit_mask, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
              std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)..., qubit);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator,
            typename LocalGate, typename... Qubits>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const global_qubit_value,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_unit_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_global_qubit,
            LocalGate&& local_gate, ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, Qubits&&... qubits)
          -> RandomAccessRange&
          {
            auto const permutated_control_qubit = permutation[control_qubit];
            if (permutated_control_qubit >= least_permutated_global_qubit)
            {
              if ((global_qubit_value bitand (StateInteger{1u} << (permutated_control_qubit - least_permutated_global_qubit))) == StateInteger{0u})
                return local_state;

              return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
                mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
                unit_control_qubit_mask, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
                std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)...);
            }

            if (permutated_control_qubit >= least_permutated_unit_qubit)
              return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
                mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
                unit_control_qubit_mask bitor (StateInteger{1u} << (permutated_control_qubit - least_permutated_unit_qubit)),
                global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
                std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)...);

            return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              unit_control_qubit_mask, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
              std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)..., control_qubit);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype,
            typename LocalGate, typename... Qubits>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const global_qubit_value,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_unit_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_global_qubit,
            LocalGate&& local_gate, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits&&... qubits)
          -> RandomAccessRange&
          {
            return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, sizeof...(Qubits) + 1u>::call(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              unit_control_qubit_mask, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
              std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)..., qubit);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype,
            typename LocalGate, typename... Qubits>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const global_qubit_value,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_unit_qubit,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_global_qubit,
            LocalGate&& local_gate, ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, Qubits&&... qubits)
          -> RandomAccessRange&
          {
            auto const permutated_control_qubit = permutation[control_qubit];
            if (permutated_control_qubit >= least_permutated_global_qubit)
            {
              if ((global_qubit_value bitand (StateInteger{1u} << (permutated_control_qubit - least_permutated_global_qubit))) == StateInteger{0u})
                return local_state;

              return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
                mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
                unit_control_qubit_mask, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
                std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)...);
            }

            if (permutated_control_qubit >= least_permutated_unit_qubit)
              return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
                mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
                unit_control_qubit_mask bitor (StateInteger{1u} << (permutated_control_qubit - least_permutated_unit_qubit)),
                global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
                std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)...);

            return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              unit_control_qubit_mask, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
              std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)..., control_qubit);
          }
        }; // struct apply_local_gate<index, num_operated_qubits>

        template <std::size_t num_operated_qubits>
        struct apply_local_gate<num_operated_qubits, num_operated_qubits>
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator,
            typename LocalGate, typename Qubit, typename... Qubits>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            LocalGate&& local_gate, Qubit&& qubit, Qubits&&... qubits)
          -> RandomAccessRange&
          {
            ::ket::mpi::utility::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment, qubit, qubits...);

            local_gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment,
              unit_control_qubit_mask, permutation[std::forward<Qubit>(qubit)], permutation[std::forward<Qubits>(qubits)]...);
            return local_state;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator,
            typename LocalGate>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>&,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            LocalGate&& local_gate)
          -> RandomAccessRange&
          { local_gate(mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask); return local_state; }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype,
            typename LocalGate, typename Qubit, typename... Qubits>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            LocalGate&& local_gate, Qubit&& qubit, Qubits&&... qubits)
          -> RandomAccessRange&
          {
            ::ket::mpi::utility::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);

            local_gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment,
              unit_control_qubit_mask, permutation[std::forward<Qubit>(qubit)], permutation[std::forward<Qubits>(qubits)]...);
            return local_state;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype,
            typename LocalGate>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const&,
            yampi::communicator const& communicator, yampi::environment const& environment,
            StateInteger const unit_control_qubit_mask, StateInteger const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const,
            LocalGate&& local_gate)
          -> RandomAccessRange&
          { local_gate(mpi_policy, parallel_policy, local_state, buffer, communicator, environment, unit_control_qubit_mask); return local_state; }
        }; // struct apply_local_gate<num_operated_qubits, num_operated_qubits>
      } // namespace apply_local_gate_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator,
        typename LocalGate, typename... Qubits>
      inline auto apply_local_gate(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        LocalGate&& local_gate, Qubits&&... qubits)
      -> RandomAccessRange&
      {
        auto const global_qubit_value = ::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment);
        auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
        auto const least_permutated_unit_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, data_block_size))));
        auto const least_permutated_global_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_nonglobal_qubits(mpi_policy, data_block_size))));

        return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<0u, sizeof...(Qubits)>::call(
          mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
          StateInteger{0u}, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
          std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype,
        typename LocalGate, typename... Qubits>
      inline auto apply_local_gate(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        LocalGate&& local_gate, Qubits&&... qubits)
      -> RandomAccessRange&
      {
        auto const global_qubit_value = ::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment);
        auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
        auto const least_permutated_unit_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, data_block_size))));
        auto const least_permutated_global_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_nonglobal_qubits(mpi_policy, data_block_size))));

        return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<0u, sizeof...(Qubits)>::call(
          mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
          StateInteger{0u}, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
          std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)...);
      }
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_APPLY_LOCAL_GATE_HPP
