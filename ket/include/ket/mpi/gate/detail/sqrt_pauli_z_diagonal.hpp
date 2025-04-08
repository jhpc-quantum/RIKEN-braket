#ifndef KET_MPI_GATE_DETAIL_SQRT_PAULI_Z_DIAGONAL_HPP
# define KET_MPI_GATE_DETAIL_SQRT_PAULI_Z_DIAGONAL_HPP

# include <vector>
# include <array>
# include <iterator>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
#   include <ket/control_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/sqrt_pauli_z.hpp>
# include <ket/gate/meta/num_control_qubits.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/gate/page/sqrt_pauli_z_diagonal.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace local
      {
        // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
        // sZ_i
        // sZ_1 (a_0 |0> + a_1 |1>) = a_0 |0> + i a_1 |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          auto const permutated_control_qubit = permutation[control_qubit];
          if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::sqrt_pauli_z1(parallel_policy, local_state, permutated_control_qubit);

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment,
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
            control_qubit);

          return local_state;
        }

        // CsZ_{cc'} or C1sZ_{cc'}
        // CsZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
        -> RandomAccessRange&
        {
          auto const permutated_control_qubit1 = permutation[control_qubit1];
          auto const permutated_control_qubit2 = permutation[control_qubit2];
          if (::ket::mpi::page::is_on_page(permutated_control_qubit1, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
              return ::ket::mpi::gate::page::sqrt_pauli_cz_2p(
                parallel_policy, local_state, permutated_control_qubit1, permutated_control_qubit2);

            return ::ket::mpi::gate::page::sqrt_pauli_cz_p(
              mpi_policy, parallel_policy, local_state, permutated_control_qubit1, permutated_control_qubit2,
              communicator.rank(environment));
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
            return ::ket::mpi::gate::page::sqrt_pauli_cz_p(
              mpi_policy, parallel_policy, local_state, permutated_control_qubit2, permutated_control_qubit1,
              communicator.rank(environment));

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment,
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
            control_qubit1, control_qubit2);

          return local_state;
        }

        // C...CsZ_{c0,c...c'} or CnsZ_{c0,c...c'}
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator,
          typename... Qubits>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
          ::ket::control<Qubits> const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubit3], permutation[control_qubits]...);

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          using std::begin;
          auto const first = begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::sqrt_pauli_z(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[control_qubit1].qubit(), permutation[control_qubit2].qubit(), permutation[control_qubit3].qubit(), permutation[control_qubits].qubit()...);

          return local_state;
        }

        // Case 2: the first argument of qubits is ket::qubit<S, B>
        // sZ_i
        // sZ_1 (a_0 |0> + a_1 |1>) = a_0 |0> + i a_1 |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::sqrt_pauli_z1(parallel_policy, local_state, permutated_qubit);

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            [](auto const, StateInteger const) { },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); });

          return local_state;
        }

        // sZZ_{ij} = sZ_i sZ_j or sZ2_{ij}
        // sZZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + i a_{01} |01> + i a_{10} |10> - a{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2)
        -> RandomAccessRange&
        {
          auto const permutated_qubit1 = permutation[qubit1];
          auto const permutated_qubit2 = permutation[qubit2];
          if (::ket::mpi::page::is_on_page(permutated_qubit1, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
              return ::ket::mpi::gate::page::sqrt_pauli_z2_2p(
                parallel_policy, local_state, permutated_qubit1, permutated_qubit2);

            return ::ket::mpi::gate::page::sqrt_pauli_z2_p(
              mpi_policy, parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
              communicator.rank(environment));
          }
          else if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
            return ::ket::mpi::gate::page::sqrt_pauli_z2_p(
              mpi_policy, parallel_policy, local_state, permutated_qubit2, permutated_qubit1,
              communicator.rank(environment));

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit1, qubit2,
            [](auto const, StateInteger const) { },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
            [](auto const iter, StateInteger const) { *iter *= real_type{-1}; });

          return local_state;
        }

        // CsZ_{tc} or C1sZ_{tc}
        // CsZ_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> + i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
              return ::ket::mpi::gate::page::sqrt_pauli_cz_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::sqrt_pauli_cz_tp(
              mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
              communicator.rank(environment));
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::sqrt_pauli_cz_cp(
              mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
              communicator.rank(environment));

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, target_qubit,
            [](auto const, StateInteger const) { },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
            control_qubit);

          return local_state;
        }

        // C...CsZ...Z_{t...t'c...c'} = C...C(sZ_t ... sZ_t')_{c...c'}, CnsZ...Z_{...}, C...CsZm_{...}, or CnsZm_{...}
        //   (sZ_1...sZ_N)_{nn} = i^f(n-1) for 1<=n<=2^N, where f(n): the number of "1" bits in n
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator,
          typename Qubit2, typename Qubit3, typename... Qubits>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[qubit1], permutation[qubit2], permutation[qubit3], permutation[qubits]...);

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          using std::begin;
          auto const first = begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::sqrt_pauli_z(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[qubit1].qubit(), permutation[qubit2].qubit(), permutation[qubit3].qubit(), permutation[qubits].qubit()...);

          return local_state;
        }
      } // namespace local

      namespace sqrt_pauli_z_detail
      {
        // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... Qubits>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
          ::ket::control<Qubits> const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);

          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
          ::ket::control<Qubits> const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);

          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);
        }

        // Case 2: the first argument of qubits is ket::qubit<S, B>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename Qubit2>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Qubit2>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename Qubit2, typename Qubit3, typename... Qubits>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, qubit1, qubit2, qubit3, qubits...);

          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2, qubit3, qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Qubit2, typename Qubit3, typename... Qubits>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2, qubit3, qubits...);

          return ::ket::mpi::gate::local::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2, qubit3, qubits...);
        }
      } // namespace sqrt_pauli_z_detail

      // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits)};
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append("sZ"), control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline auto sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits)};
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append("sZ"), control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, control_qubit, control_qubits...);
      }

      // Case 2: the first argument of qubits is ket::qubit<S, B>
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append("s").append(sizeof...(Qubits) + 1u - num_control_qubits, 'Z'), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline auto sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string(num_control_qubits, 'C').append("s").append(sizeof...(Qubits) + 1u - num_control_qubits, 'Z'), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      namespace local
      {
        // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
        // sZ+_i
        // sZ+_1 (a_0 |0> + a_1 |1>) = a_0 |0> - i a_1 |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          auto const permutated_control_qubit = permutation[control_qubit];
          if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::adj_sqrt_pauli_z1(parallel_policy, local_state, permutated_control_qubit);

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment,
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
            control_qubit);

          return local_state;
        }

        // CsZ+_{cc'} or C1sZ+_{cc'}
        // CsZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
        -> RandomAccessRange&
        {
          auto const permutated_control_qubit1 = permutation[control_qubit1];
          auto const permutated_control_qubit2 = permutation[control_qubit2];
          if (::ket::mpi::page::is_on_page(permutated_control_qubit1, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
              return ::ket::mpi::gate::page::adj_sqrt_pauli_cz_2p(
                parallel_policy, local_state, permutated_control_qubit1, permutated_control_qubit2);

            return ::ket::mpi::gate::page::adj_sqrt_pauli_cz_p(
              mpi_policy, parallel_policy, local_state, permutated_control_qubit1, permutated_control_qubit2,
              communicator.rank(environment));
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
            return ::ket::mpi::gate::page::adj_sqrt_pauli_cz_p(
              mpi_policy, parallel_policy, local_state, permutated_control_qubit2, permutated_control_qubit1,
              communicator.rank(environment));

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment,
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
            control_qubit1, control_qubit2);

          return local_state;
        }

        // C...CsZ+_{c0,c...c'} or CnsZ+_{c0,c...c'}
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator,
          typename... Qubits>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
          ::ket::control<Qubits> const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[control_qubit1], permutation[control_qubit2], permutation[control_qubit3], permutation[control_qubits]...);

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          using std::begin;
          auto const first = begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::adj_sqrt_pauli_z(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[control_qubit1].qubit(), permutation[control_qubit2].qubit(), permutation[control_qubit3].qubit(), permutation[control_qubits].qubit()...);

          return local_state;
        }

        // Case 2: the first argument of qubits is ket::qubit<S, B>
        // sZ+_i
        // sZ+_1 (a_0 |0> + a_1 |1>) = a_0 |0> - i a_1 |1>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::adj_sqrt_pauli_z1(parallel_policy, local_state, permutated_qubit);

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            [](auto const, StateInteger const) { },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); });

          return local_state;
        }

        // sZZ+_{ij} = sZ_i sZ_j or sZ2+_{ij}
        // sZZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> - i a_{01} |01> - i a_{10} |10> - a{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2)
        -> RandomAccessRange&
        {
          auto const permutated_qubit1 = permutation[qubit1];
          auto const permutated_qubit2 = permutation[qubit2];
          if (::ket::mpi::page::is_on_page(permutated_qubit1, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
              return ::ket::mpi::gate::page::adj_sqrt_pauli_z2_2p(
                parallel_policy, local_state, permutated_qubit1, permutated_qubit2);

            return ::ket::mpi::gate::page::adj_sqrt_pauli_z2_p(
              mpi_policy, parallel_policy, local_state, permutated_qubit1, permutated_qubit2,
              communicator.rank(environment));
          }
          else if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
            return ::ket::mpi::gate::page::adj_sqrt_pauli_z2_p(
              mpi_policy, parallel_policy, local_state, permutated_qubit2, permutated_qubit1,
              communicator.rank(environment));

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          using real_type = ::ket::utility::meta::real_t<complex_type>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit1, qubit2,
            [](auto const, StateInteger const) { },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
            [](auto const iter, StateInteger const) { *iter *= real_type{-1}; });

          return local_state;
        }

        // CsZ+_{tc} or C1sZ+_{tc}
        // CsZ+_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + a_{10} |10> - i a_{11} |11>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          auto const permutated_target_qubit = permutation[target_qubit];
          auto const permutated_control_qubit = permutation[control_qubit];
          if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
              return ::ket::mpi::gate::page::adj_sqrt_pauli_cz_tcp(
                parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);

            return ::ket::mpi::gate::page::adj_sqrt_pauli_cz_tp(
              mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
              communicator.rank(environment));
          }
          else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
            return ::ket::mpi::gate::page::adj_sqrt_pauli_cz_cp(
              mpi_policy, parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
              communicator.rank(environment));

          using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, target_qubit,
            [](auto const, StateInteger const) { },
            [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
            control_qubit);

          return local_state;
        }

        // C...CsZ...Z+_{t...t'c...c'} = C...C(sZ+_t ... sZ+_t')_{c...c'}, CnsZ...Z+_{...}, C...CsZm+_{...}, or CnsZm+_{...}
        //   (sZ+_1...sZ+_N)_{nn} = (-i)^f(n-1) for 1<=n<=2^N, where f(n): the number of "1" bits in n
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator,
          typename Qubit2, typename Qubit3, typename... Qubits>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::gate::detail::assert_all_qubits_are_local(
            mpi_policy, local_state, communicator, environment,
            permutation[qubit1], permutation[qubit2], permutation[qubit3], permutation[qubits]...);

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          using std::begin;
          auto const first = begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::adj_sqrt_pauli_z(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[qubit1].qubit(), permutation[qubit2].qubit(), permutation[qubit3].qubit(), permutation[qubits].qubit()...);

          return local_state;
        }
      } // namespace local

      namespace sqrt_pauli_z_detail
      {
        // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename... Qubits>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
          ::ket::control<Qubits> const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);

          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2,
          ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit3,
          ::ket::control<Qubits> const... control_qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);

          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubit1, control_qubit2, control_qubit3, control_qubits...);
        }

        // Case 2: the first argument of qubits is ket::qubit<S, B>
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename Qubit2>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Qubit2>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >&,
          yampi::datatype_base<DerivedDatatype> const&,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename Qubit2, typename Qubit3, typename... Qubits>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, communicator, environment, qubit1, qubit2, qubit3, qubits...);

          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2, qubit3, qubits...);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Qubit2, typename Qubit3, typename... Qubits>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ::ket::qubit<StateInteger, BitInteger> const qubit1, Qubit2 const qubit2, Qubit3 const qubit3, Qubits const... qubits)
        -> RandomAccessRange&
        {
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, permutation, buffer, datatype, communicator, environment, qubit1, qubit2, qubit3, qubits...);

          return ::ket::mpi::gate::local::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, communicator, environment, qubit1, qubit2, qubit3, qubits...);
        }
      } // namespace sqrt_pauli_z_detail

      // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits)};
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(num_control_qubits, 'C').append("sZ)"), control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = BitInteger{sizeof...(Qubits)};
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(num_control_qubits, 'C').append("sZ)"), control_qubit, control_qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, control_qubit, control_qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit, ::ket::control<Qubits> const... control_qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, control_qubit, control_qubits...);
      }

      // Case 2: the first argument of qubits is ket::qubit<S, B>
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(num_control_qubits, 'C').append("s").append(sizeof...(Qubits) + 1u - num_control_qubits, 'Z').append(")"), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        constexpr auto num_control_qubits = ::ket::gate::meta::num_control_qubits<BitInteger, Qubits...>::value;
        ::ket::mpi::utility::log_with_time_guard<char> print{
          ::ket::mpi::gate::detail::append_qubits_string(std::string{"Adj("}.append(num_control_qubits, 'C').append("s").append(sizeof...(Qubits) + 1u - num_control_qubits, 'Z').append(")"), qubit, qubits...),
          environment};

        return ::ket::mpi::gate::sqrt_pauli_z_detail::sqrt_pauli_z(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, communicator, environment, qubit, qubits...);
      }

      template <
        typename ParallelPolicy,
        typename RandomAccessRange, typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename... Qubits>
      inline auto adj_sqrt_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> RandomAccessRange&
      {
        return ::ket::mpi::gate::adj_sqrt_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, permutation, buffer, datatype, communicator, environment, qubit, qubits...);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_SQRT_PAULI_Z_DIAGONAL_HPP
