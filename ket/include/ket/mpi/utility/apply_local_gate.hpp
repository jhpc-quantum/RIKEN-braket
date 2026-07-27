#ifndef KET_MPI_UTILITY_APPLY_LOCAL_GATE_HPP
# define KET_MPI_UTILITY_APPLY_LOCAL_GATE_HPP

# include <cstddef>
# include <array>
# include <vector>
# include <iterator>
# include <utility>

# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>
# include <boost/range/adaptor/transformed.hpp>

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
            return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<index + 1u, num_operated_qubits>::call(
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
        auto const global_qubit_value = static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment));
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
        auto const global_qubit_value = static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment));
        auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
        auto const least_permutated_unit_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, data_block_size))));
        auto const least_permutated_global_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_nonglobal_qubits(mpi_policy, data_block_size))));

        return ::ket::mpi::utility::apply_local_gate_detail::apply_local_gate<0u, sizeof...(Qubits)>::call(
          mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
          StateInteger{0u}, global_qubit_value, least_permutated_unit_qubit, least_permutated_global_qubit,
          std::forward<LocalGate>(local_gate), std::forward<Qubits>(qubits)...);
      }


      namespace runtime
      {
        namespace ranges
        {
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator,
            typename LocalGate, typename Qubits, typename ControlQubits>
          inline auto apply_local_gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            LocalGate&& local_gate, Qubits const& qubits, ControlQubits const& control_qubits)
          -> RandomAccessRange&
          {
            using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubits>;

            auto const global_qubit_value = static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment));
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
            auto const least_permutated_unit_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, data_block_size))));
            auto const least_permutated_global_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_nonglobal_qubits(mpi_policy, data_block_size))));

            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<BitInteger>(std::distance(begin(control_qubits), end(control_qubits)));
            auto local_control_qubits = std::vector<control_qubit_type>{};
            local_control_qubits.reserve(num_control_qubits);

            auto unit_control_qubit_mask = StateInteger{0u};
            for (auto const control_qubit: control_qubits)
            {
              auto const permutated_control_qubit = permutation[control_qubit];
              if (permutated_control_qubit >= least_permutated_global_qubit)
              {
                if ((global_qubit_value bitand (StateInteger{1u} << (permutated_control_qubit - least_permutated_global_qubit))) == StateInteger{0u})
                  return local_state;

                continue;
              }

              if (permutated_control_qubit >= least_permutated_unit_qubit)
              {
                unit_control_qubit_mask |= StateInteger{1u} << (permutated_control_qubit - least_permutated_unit_qubit);
                continue;
              }

              local_control_qubits.push_back(control_qubit);
            }

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                qubits,
                local_control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            local_gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment,
              unit_control_qubit_mask,
              qubits | boost::adaptors::transformed(
                [&permutation](qubit_type const qubit) { return permutation[qubit]; }),
              local_control_qubits | boost::adaptors::transformed(
                [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; }));

            return local_state;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator,
            typename LocalGate, typename Qubits>
          inline auto apply_local_gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            LocalGate&& local_gate, Qubits const& qubits)
          -> RandomAccessRange&
          {
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment, qubits);

            using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;
            local_gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment,
              qubits | boost::adaptors::transformed(
                [&permutation](qubit_type const qubit) { return permutation[qubit]; }));

            return local_state;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype,
            typename LocalGate, typename Qubits, typename ControlQubits>
          inline auto apply_local_gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            LocalGate&& local_gate, Qubits const& qubits, ControlQubits const& control_qubits)
          -> RandomAccessRange&
          {
            using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubits>;

            auto const global_qubit_value = static_cast<StateInteger>(::ket::mpi::utility::policy::global_qubit_value(mpi_policy, communicator, environment));
            auto const data_block_size = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
            auto const least_permutated_unit_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, data_block_size))));
            auto const least_permutated_global_qubit = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(static_cast<BitInteger>(::ket::mpi::utility::policy::num_nonglobal_qubits(mpi_policy, data_block_size))));

            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<BitInteger>(std::distance(begin(control_qubits), end(control_qubits)));
            auto local_control_qubits = std::vector<control_qubit_type>{};
            local_control_qubits.reserve(num_control_qubits);

            auto unit_control_qubit_mask = StateInteger{0u};
            for (auto const control_qubit: control_qubits)
            {
              auto const permutated_control_qubit = permutation[control_qubit];
              if (permutated_control_qubit >= least_permutated_global_qubit)
              {
                if ((global_qubit_value bitand (StateInteger{1u} << (permutated_control_qubit - least_permutated_global_qubit))) == StateInteger{0u})
                  return local_state;

                continue;
              }

              if (permutated_control_qubit >= least_permutated_unit_qubit)
              {
                unit_control_qubit_mask |= StateInteger{1u} << (permutated_control_qubit - least_permutated_unit_qubit);
                continue;
              }

              local_control_qubits.push_back(control_qubit);
            }

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                qubits,
                local_control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            local_gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment,
              unit_control_qubit_mask,
              qubits | boost::adaptors::transformed(
                [&permutation](qubit_type const qubit) { return permutation[qubit]; }),
              local_control_qubits | boost::adaptors::transformed(
                [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; }));

            return local_state;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype,
            typename LocalGate, typename Qubits>
          inline auto apply_local_gate(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            LocalGate&& local_gate, Qubits const& qubits)
          -> RandomAccessRange&
          {
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment, qubits);

            using qubit_type = ::ket::utility::meta::range_value_t<Qubits>;
            local_gate(
              mpi_policy, parallel_policy, local_state, buffer, communicator, environment,
              qubits | boost::adaptors::transformed(
                [&permutation](qubit_type const qubit) { return permutation[qubit]; }));

            return local_state;
          }
        } // namespace ranges

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator,
          typename LocalGate, typename QubitIterator, typename ControlQubitIterator>
        inline auto apply_local_gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          LocalGate&& local_gate,
          QubitIterator const qubit_first, QubitIterator const qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::utility::runtime::ranges::apply_local_gate(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment,
            std::forward<LocalGate>(local_gate),
            boost::make_iterator_range(qubit_first, qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator,
          typename LocalGate, typename QubitIterator>
        inline auto apply_local_gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          LocalGate&& local_gate,
          QubitIterator const qubit_first, QubitIterator const qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::utility::runtime::ranges::apply_local_gate(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment,
            std::forward<LocalGate>(local_gate),
            boost::make_iterator_range(qubit_first, qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype,
          typename LocalGate, typename QubitIterator, typename ControlQubitIterator>
        inline auto apply_local_gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          LocalGate&& local_gate,
          QubitIterator const qubit_first, QubitIterator const qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::utility::runtime::ranges::apply_local_gate(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            std::forward<LocalGate>(local_gate),
            boost::make_iterator_range(qubit_first, qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype,
          typename LocalGate, typename QubitIterator>
        inline auto apply_local_gate(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          LocalGate&& local_gate,
          QubitIterator const qubit_first, QubitIterator const qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::utility::runtime::ranges::apply_local_gate(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment,
            std::forward<LocalGate>(local_gate),
            boost::make_iterator_range(qubit_first, qubit_last));
        }
      } // namespace runtime
    } // namespace utility
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_UTILITY_APPLY_LOCAL_GATE_HPP
