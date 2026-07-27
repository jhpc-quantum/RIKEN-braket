#ifndef KET_MPI_GATE_DETAIL_SQRT_PAULI_Z_RUNTIME_HPP
# define KET_MPI_GATE_DETAIL_SQRT_PAULI_Z_RUNTIME_HPP

# include <array>
# include <algorithm>
# include <iterator>
# include <type_traits>
# include <utility>
# include <vector>

# include <yampi/communicator.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/environment.hpp>

# include <boost/range/adaptor/transformed.hpp>
# include <boost/range/iterator_range.hpp>
# include <boost/range/join.hpp>

# include <ket/control.hpp>
# include <ket/gate/sqrt_pauli_z.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/page/any_on_page.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/qubit.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/ranges.hpp>

namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace runtime
      {
        namespace local
        {
          namespace dispatch
          {
            template <typename LocalState>
            struct transpage_sqrt_pauli_z
            {
              template <typename ParallelPolicy, typename RandomAccessRange, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }

              template <typename ParallelPolicy, typename RandomAccessRange, typename PermutatedTargetQubitsRange, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                PermutatedTargetQubitsRange const& permutated_target_qubits,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_sqrt_pauli_z<LocalState>

            template <typename LocalState>
            struct transpage_adj_sqrt_pauli_z
            {
              template <typename ParallelPolicy, typename RandomAccessRange, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }

              template <typename ParallelPolicy, typename RandomAccessRange, typename PermutatedTargetQubitsRange, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                PermutatedTargetQubitsRange const& permutated_target_qubits,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_adj_sqrt_pauli_z<LocalState>
          } // namespace dispatch

          // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment, permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(local_state, permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_sqrt_pauli_z<local_state_type>::call(
                parallel_policy, local_state, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::sqrt_pauli_z(
                  parallel_policy, first, last,
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }

          // Case 2: the first argument of qubits is ket::qubit<S, B>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using target_qubit_type = ::ket::utility::meta::range_value_t<TargetQubitsRange>;
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_target_qubits
              = target_qubits | boost::adaptors::transformed(
                  [&permutation](target_qubit_type const target_qubit) { return permutation[target_qubit]; });
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment, permutated_target_qubits, permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(local_state, permutated_target_qubits, permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_sqrt_pauli_z<local_state_type>::call(
                parallel_policy, local_state, permutated_target_qubits, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, permutated_target_qubits, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::sqrt_pauli_z(
                  parallel_policy, first, last,
                  permutated_target_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_target_qubits)> const permutated_target_qubit)
                    { return permutated_target_qubit.qubit(); }),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment, permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(local_state, permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_adj_sqrt_pauli_z<local_state_type>::call(
                parallel_policy, local_state, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_z(
                  parallel_policy, first, last,
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using target_qubit_type = ::ket::utility::meta::range_value_t<TargetQubitsRange>;
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_target_qubits
              = target_qubits | boost::adaptors::transformed(
                  [&permutation](target_qubit_type const target_qubit) { return permutation[target_qubit]; });
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment, permutated_target_qubits, permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(local_state, permutated_target_qubits, permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_adj_sqrt_pauli_z<local_state_type>::call(
                parallel_policy, local_state, permutated_target_qubits, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, permutated_target_qubits, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::adj_sqrt_pauli_z(
                  parallel_policy, first, last,
                  permutated_target_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_target_qubits)> const permutated_target_qubit)
                    { return permutated_target_qubit.qubit(); }),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }
        } // namespace local

        namespace sqrt_pauli_z_detail
        {
# ifdef KET_USE_DIAGONAL_LOOP
          // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename ControlQubitsRange>
          inline auto maybe_apply_diagonal_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            auto const control_qubit_first = begin(control_qubits);
            auto const num_control_qubits = std::distance(control_qubit_first, end(control_qubits));
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;

            if (num_control_qubits == 0)
            {
              ::ket::mpi::utility::diagonal_loop(
                mpi_policy, parallel_policy,
                local_state, permutation, communicator, environment,
                [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); });
              return true;
            }

            if (num_control_qubits == 1)
            {
              auto const control_qubit = *control_qubit_first;
              auto const permutated_control_qubit = permutation[control_qubit];
              if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                ::ket::mpi::gate::page::sqrt_pauli_z1(parallel_policy, local_state, permutated_control_qubit);
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment,
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
                  control_qubit);
              return true;
            }

            if (num_control_qubits == 2)
            {
              auto const control_qubit1 = *control_qubit_first;
              auto const control_qubit2 = *std::next(control_qubit_first);
              auto const permutated_control_qubit1 = permutation[control_qubit1];
              auto const permutated_control_qubit2 = permutation[control_qubit2];
              if (::ket::mpi::page::is_on_page(permutated_control_qubit1, local_state))
              {
                if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
                  ::ket::mpi::gate::page::sqrt_pauli_cz_2p(
                    parallel_policy, local_state, permutated_control_qubit1, permutated_control_qubit2);
                else
                  ::ket::mpi::gate::page::sqrt_pauli_cz_p(
                    mpi_policy, parallel_policy, local_state,
                    permutated_control_qubit1, permutated_control_qubit2,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
                ::ket::mpi::gate::page::sqrt_pauli_cz_p(
                  mpi_policy, parallel_policy, local_state,
                  permutated_control_qubit2, permutated_control_qubit1,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment,
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
                  control_qubit1, control_qubit2);
              return true;
            }

            return false;
          }

          // Case 2: the first argument of qubits is ket::qubit<S, B>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto maybe_apply_diagonal_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits,
            ControlQubitsRange const& control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            auto const target_qubit_first = begin(target_qubits);
            auto const control_qubit_first = begin(control_qubits);
            auto const num_target_qubits = std::distance(target_qubit_first, end(target_qubits));
            auto const num_control_qubits = std::distance(control_qubit_first, end(control_qubits));
            if (num_target_qubits + num_control_qubits > 2)
              return false;

            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using real_type = ::ket::utility::meta::real_t<complex_type>;

            if (num_target_qubits == 1 and num_control_qubits == 0)
            {
              auto const target_qubit = *target_qubit_first;
              auto const permutated_target_qubit = permutation[target_qubit];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
                ::ket::mpi::gate::page::sqrt_pauli_z1(parallel_policy, local_state, permutated_target_qubit);
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit,
                  [](auto const, StateInteger const) { },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); });
              return true;
            }

            if (num_target_qubits == 2 and num_control_qubits == 0)
            {
              auto const target_qubit1 = *target_qubit_first;
              auto const target_qubit2 = *std::next(target_qubit_first);
              auto const permutated_target_qubit1 = permutation[target_qubit1];
              auto const permutated_target_qubit2 = permutation[target_qubit2];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit1, local_state))
              {
                if (::ket::mpi::page::is_on_page(permutated_target_qubit2, local_state))
                  ::ket::mpi::gate::page::sqrt_pauli_z2_2p(parallel_policy, local_state, permutated_target_qubit1, permutated_target_qubit2);
                else
                  ::ket::mpi::gate::page::sqrt_pauli_z2_p(
                    mpi_policy, parallel_policy, local_state,
                    permutated_target_qubit1, permutated_target_qubit2,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_target_qubit2, local_state))
                ::ket::mpi::gate::page::sqrt_pauli_z2_p(
                  mpi_policy, parallel_policy, local_state,
                  permutated_target_qubit2, permutated_target_qubit1,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit1, target_qubit2,
                  [](auto const, StateInteger const) { },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
                  [](auto const iter, StateInteger const) { *iter *= real_type{-1}; });
              return true;
            }

            if (num_target_qubits == 1 and num_control_qubits == 1)
            {
              auto const target_qubit = *target_qubit_first;
              auto const control_qubit = *control_qubit_first;
              auto const permutated_target_qubit = permutation[target_qubit];
              auto const permutated_control_qubit = permutation[control_qubit];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
              {
                if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                  ::ket::mpi::gate::page::sqrt_pauli_cz_tcp(
                    parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
                else
                  ::ket::mpi::gate::page::sqrt_pauli_cz_tp(
                    mpi_policy, parallel_policy, local_state,
                    permutated_target_qubit, permutated_control_qubit,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                ::ket::mpi::gate::page::sqrt_pauli_cz_cp(
                  mpi_policy, parallel_policy, local_state,
                  permutated_target_qubit, permutated_control_qubit,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit,
                  [](auto const, StateInteger const) { },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::imaginary_unit<complex_type>(); },
                  control_qubit);
              return true;
            }

            return false;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename ControlQubitsRange>
          inline auto maybe_apply_diagonal_adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            auto const control_qubit_first = begin(control_qubits);
            auto const num_control_qubits = std::distance(control_qubit_first, end(control_qubits));
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;

            if (num_control_qubits == 0)
            {
              ::ket::mpi::utility::diagonal_loop(
                mpi_policy, parallel_policy,
                local_state, permutation, communicator, environment,
                [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); });
              return true;
            }

            if (num_control_qubits == 1)
            {
              auto const control_qubit = *control_qubit_first;
              auto const permutated_control_qubit = permutation[control_qubit];
              if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                ::ket::mpi::gate::page::adj_sqrt_pauli_z1(parallel_policy, local_state, permutated_control_qubit);
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment,
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
                  control_qubit);
              return true;
            }

            if (num_control_qubits == 2)
            {
              auto const control_qubit1 = *control_qubit_first;
              auto const control_qubit2 = *std::next(control_qubit_first);
              auto const permutated_control_qubit1 = permutation[control_qubit1];
              auto const permutated_control_qubit2 = permutation[control_qubit2];
              if (::ket::mpi::page::is_on_page(permutated_control_qubit1, local_state))
              {
                if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
                  ::ket::mpi::gate::page::adj_sqrt_pauli_cz_2p(
                    parallel_policy, local_state, permutated_control_qubit1, permutated_control_qubit2);
                else
                  ::ket::mpi::gate::page::adj_sqrt_pauli_cz_p(
                    mpi_policy, parallel_policy, local_state,
                    permutated_control_qubit1, permutated_control_qubit2,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
                ::ket::mpi::gate::page::adj_sqrt_pauli_cz_p(
                  mpi_policy, parallel_policy, local_state,
                  permutated_control_qubit2, permutated_control_qubit1,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment,
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
                  control_qubit1, control_qubit2);
              return true;
            }

            return false;
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto maybe_apply_diagonal_adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits,
            ControlQubitsRange const& control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            auto const target_qubit_first = begin(target_qubits);
            auto const control_qubit_first = begin(control_qubits);
            auto const num_target_qubits = std::distance(target_qubit_first, end(target_qubits));
            auto const num_control_qubits = std::distance(control_qubit_first, end(control_qubits));
            if (num_target_qubits + num_control_qubits > 2)
              return false;

            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using real_type = ::ket::utility::meta::real_t<complex_type>;

            if (num_target_qubits == 1 and num_control_qubits == 0)
            {
              auto const target_qubit = *target_qubit_first;
              auto const permutated_target_qubit = permutation[target_qubit];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
                ::ket::mpi::gate::page::adj_sqrt_pauli_z1(parallel_policy, local_state, permutated_target_qubit);
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit,
                  [](auto const, StateInteger const) { },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); });
              return true;
            }

            if (num_target_qubits == 2 and num_control_qubits == 0)
            {
              auto const target_qubit1 = *target_qubit_first;
              auto const target_qubit2 = *std::next(target_qubit_first);
              auto const permutated_target_qubit1 = permutation[target_qubit1];
              auto const permutated_target_qubit2 = permutation[target_qubit2];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit1, local_state))
              {
                if (::ket::mpi::page::is_on_page(permutated_target_qubit2, local_state))
                  ::ket::mpi::gate::page::adj_sqrt_pauli_z2_2p(parallel_policy, local_state, permutated_target_qubit1, permutated_target_qubit2);
                else
                  ::ket::mpi::gate::page::adj_sqrt_pauli_z2_p(
                    mpi_policy, parallel_policy, local_state,
                    permutated_target_qubit1, permutated_target_qubit2,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_target_qubit2, local_state))
                ::ket::mpi::gate::page::adj_sqrt_pauli_z2_p(
                  mpi_policy, parallel_policy, local_state,
                  permutated_target_qubit2, permutated_target_qubit1,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit1, target_qubit2,
                  [](auto const, StateInteger const) { },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
                  [](auto const iter, StateInteger const) { *iter *= real_type{-1}; });
              return true;
            }

            if (num_target_qubits == 1 and num_control_qubits == 1)
            {
              auto const target_qubit = *target_qubit_first;
              auto const control_qubit = *control_qubit_first;
              auto const permutated_target_qubit = permutation[target_qubit];
              auto const permutated_control_qubit = permutation[control_qubit];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
              {
                if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                  ::ket::mpi::gate::page::adj_sqrt_pauli_cz_tcp(
                    parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit);
                else
                  ::ket::mpi::gate::page::adj_sqrt_pauli_cz_tp(
                    mpi_policy, parallel_policy, local_state,
                    permutated_target_qubit, permutated_control_qubit,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                ::ket::mpi::gate::page::adj_sqrt_pauli_cz_cp(
                  mpi_policy, parallel_policy, local_state,
                  permutated_target_qubit, permutated_control_qubit,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit,
                  [](auto const, StateInteger const) { },
                  [](auto const iter, StateInteger const) { *iter *= ::ket::utility::minus_imaginary_unit<complex_type>(); },
                  control_qubit);
              return true;
            }

            return false;
          }
# endif // KET_USE_DIAGONAL_LOOP

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); }));

            return ::ket::mpi::gate::runtime::local::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); }));

            return ::ket::mpi::gate::runtime::local::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_adj_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); }));

            return ::ket::mpi::gate::runtime::local::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_adj_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); }));

            return ::ket::mpi::gate::runtime::local::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_adj_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::sqrt_pauli_z_detail::maybe_apply_diagonal_adj_sqrt_pauli_z(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                target_qubits,
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, target_qubits, control_qubits);
          }
        } // namespace sqrt_pauli_z_detail

        namespace ranges
        {
          // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("sZ"),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("sZ"),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, control_qubits);
          }

          // Case 2: the first argument of qubits is ket::qubit<S, B>
          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string(num_control_qubits, 'C').append("sZ"), target_qubits, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string(num_control_qubits, 'C').append("sZ"), target_qubits, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits)
          -> std::enable_if_t<
               ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
                 ::ket::utility::meta::range_value_t<TargetQubitsRange> >::value,
               RandomAccessRange& >
          {
            using target_qubit_type = ::ket::utility::meta::range_value_t<TargetQubitsRange>;
            using control_qubit_type = ::ket::control<target_qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitsRange>
          inline auto sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits)
          -> std::enable_if_t<
               ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
                 ::ket::utility::meta::range_value_t<TargetQubitsRange> >::value,
               RandomAccessRange& >
          {
            using target_qubit_type = ::ket::utility::meta::range_value_t<TargetQubitsRange>;
            using control_qubit_type = ::ket::control<target_qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string{"Adj("}.append(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("sZ)"),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            ControlQubitsRange const& control_qubits)
          -> std::enable_if_t<
               ::ket::meta::is_control_cvref<::ket::utility::meta::range_value_t<ControlQubitsRange> >::value,
               RandomAccessRange& >
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string{"Adj("}.append(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("sZ)"),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string{"Adj("}.append(num_control_qubits, 'C').append("sZ)"), target_qubits, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitsRange, typename ControlQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                std::string{"Adj("}.append(num_control_qubits, 'C').append("sZ)"), target_qubits, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::sqrt_pauli_z_detail::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits)
          -> std::enable_if_t<
               ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
                 ::ket::utility::meta::range_value_t<TargetQubitsRange> >::value,
               RandomAccessRange& >
          {
            using target_qubit_type = ::ket::utility::meta::range_value_t<TargetQubitsRange>;
            using control_qubit_type = ::ket::control<target_qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, target_qubits, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitsRange>
          inline auto adj_sqrt_pauli_z(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            TargetQubitsRange const& target_qubits)
          -> std::enable_if_t<
               ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
                 ::ket::utility::meta::range_value_t<TargetQubitsRange> >::value,
               RandomAccessRange& >
          {
            using target_qubit_type = ::ket::utility::meta::range_value_t<TargetQubitsRange>;
            using control_qubit_type = ::ket::control<target_qubit_type>;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, target_qubits, control_qubits);
          }
        } // namespace ranges

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename ControlQubitIterator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> std::enable_if_t<
             ::ket::meta::is_control_cvref<typename std::iterator_traits<ControlQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitIterator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> std::enable_if_t<
             ::ket::meta::is_control_cvref<typename std::iterator_traits<ControlQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitIterator, typename ControlQubitIterator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitIterator, typename ControlQubitIterator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitIterator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last)
        -> std::enable_if_t<
             ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
               typename std::iterator_traits<TargetQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitIterator>
        inline auto sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last)
        -> std::enable_if_t<
             ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
               typename std::iterator_traits<TargetQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename ControlQubitIterator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> std::enable_if_t<
             ::ket::meta::is_control_cvref<typename std::iterator_traits<ControlQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename ControlQubitIterator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> std::enable_if_t<
             ::ket::meta::is_control_cvref<typename std::iterator_traits<ControlQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitIterator, typename ControlQubitIterator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitIterator, typename ControlQubitIterator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last),
            boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename TargetQubitIterator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last)
        -> std::enable_if_t<
             ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
               typename std::iterator_traits<TargetQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename TargetQubitIterator>
        inline auto adj_sqrt_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          TargetQubitIterator const target_qubit_first, TargetQubitIterator const target_qubit_last)
        -> std::enable_if_t<
             ::ket::gate::runtime::sqrt_pauli_z_detail::is_qubit<
               typename std::iterator_traits<TargetQubitIterator>::value_type>::value,
             RandomAccessRange& >
        {
          return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            boost::make_iterator_range(target_qubit_first, target_qubit_last));
        }

        namespace ranges
        {
          template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto sqrt_pauli_z(
            ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
              ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto sqrt_pauli_z(
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::sqrt_pauli_z(
              ::ket::mpi::utility::policy::make_simple_mpi(),
              ::ket::utility::policy::make_sequential(),
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto adj_sqrt_pauli_z(
            ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
              ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto adj_sqrt_pauli_z(
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::adj_sqrt_pauli_z(
              ::ket::mpi::utility::policy::make_simple_mpi(),
              ::ket::utility::policy::make_sequential(),
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }
        } // namespace ranges

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto sqrt_pauli_z(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::sqrt_pauli_z(
            ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto sqrt_pauli_z(
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::sqrt_pauli_z(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            ::ket::utility::policy::make_sequential(),
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto adj_sqrt_pauli_z(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::adj_sqrt_pauli_z(
            ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto adj_sqrt_pauli_z(
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::adj_sqrt_pauli_z(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            ::ket::utility::policy::make_sequential(),
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }
      } // namespace runtime
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_DETAIL_SQRT_PAULI_Z_RUNTIME_HPP
