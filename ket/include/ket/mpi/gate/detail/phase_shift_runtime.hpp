#ifndef KET_MPI_GATE_DETAIL_PHASE_SHIFT_RUNTIME_HPP
# define KET_MPI_GATE_DETAIL_PHASE_SHIFT_RUNTIME_HPP

# include <array>
# include <complex>
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
# include <ket/gate/phase_shift.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/gate/detail/assert_all_qubits_are_local.hpp>
# include <ket/mpi/page/any_on_page.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/qubit.hpp>
# include <ket/utility/exp_i.hpp>
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
            struct transpage_phase_shift_coeff
            {
              template <typename ParallelPolicy, typename RandomAccessRange, typename Complex, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                Complex const& phase_coefficient,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }

              template <
                typename ParallelPolicy, typename RandomAccessRange, typename Complex,
                typename StateInteger, typename BitInteger, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy,
                RandomAccessRange& local_state,
                Complex const& phase_coefficient,
                ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_phase_shift_coeff<LocalState>
          } // namespace dispatch

          // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
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
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_phase_shift_coeff<local_state_type>::call(
                parallel_policy, local_state, phase_coefficient, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, &phase_coefficient, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::phase_shift_coeff(
                  parallel_policy, first, last, phase_coefficient,
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }

          // Case 2: the first argument of qubits is ket::qubit<S, B>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_target_qubit = permutation[target_qubit];
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment,
              boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
              permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(
                  local_state,
                  boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
                  permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_phase_shift_coeff<local_state_type>::call(
                parallel_policy, local_state, phase_coefficient, permutated_target_qubit, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, &phase_coefficient, permutated_target_qubit, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::phase_shift_coeff(
                  parallel_policy, first, last, phase_coefficient,
                  permutated_target_qubit.qubit(),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }
        } // namespace local

        namespace phase_shift_detail
        {
# ifdef KET_USE_DIAGONAL_LOOP
          // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename Complex, typename ControlQubitsRange>
          inline auto maybe_apply_diagonal_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ControlQubitsRange const& control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            auto const control_qubit_first = begin(control_qubits);
            auto const num_control_qubits = std::distance(control_qubit_first, end(control_qubits));

            if (num_control_qubits == 0)
            {
              ::ket::mpi::utility::diagonal_loop(
                mpi_policy, parallel_policy,
                local_state, permutation, communicator, environment,
                [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; });
              return true;
            }

            if (num_control_qubits == 1)
            {
              auto const control_qubit = *control_qubit_first;
              auto const permutated_control_qubit = permutation[control_qubit];
              if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                ::ket::mpi::gate::page::phase_shift_coeff(
                  parallel_policy, local_state, phase_coefficient, permutated_control_qubit);
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment,
                  [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; },
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
                  ::ket::mpi::gate::page::cphase_shift_coeff_2p(
                    parallel_policy, local_state,
                    phase_coefficient, permutated_control_qubit1, permutated_control_qubit2);
                else
                  ::ket::mpi::gate::page::cphase_shift_coeff_p(
                    mpi_policy, parallel_policy, local_state,
                    phase_coefficient, permutated_control_qubit1, permutated_control_qubit2,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_control_qubit2, local_state))
                ::ket::mpi::gate::page::cphase_shift_coeff_p(
                  mpi_policy, parallel_policy, local_state,
                  phase_coefficient, permutated_control_qubit2, permutated_control_qubit1,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment,
                  [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; },
                  control_qubit1, control_qubit2);
              return true;
            }

            return false;
          }

          // Case 2: the first argument of qubits is ket::qubit<S, B>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename Complex, typename ControlQubitsRange>
          inline auto maybe_apply_diagonal_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> bool
          {
            using std::begin;
            using std::end;
            auto const control_qubit_first = begin(control_qubits);
            auto const num_control_qubits = std::distance(control_qubit_first, end(control_qubits));

            if (num_control_qubits == 0)
            {
              auto const permutated_target_qubit = permutation[target_qubit];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
                ::ket::mpi::gate::page::phase_shift_coeff(
                  parallel_policy, local_state, phase_coefficient, permutated_target_qubit);
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit,
                  [](auto const, StateInteger const) { },
                  [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; });
              return true;
            }

            if (num_control_qubits == 1)
            {
              auto const control_qubit = *control_qubit_first;
              auto const permutated_target_qubit = permutation[target_qubit];
              auto const permutated_control_qubit = permutation[control_qubit];
              if (::ket::mpi::page::is_on_page(permutated_target_qubit, local_state))
              {
                if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                  ::ket::mpi::gate::page::cphase_shift_coeff_tcp(
                    parallel_policy, local_state,
                    phase_coefficient, permutated_target_qubit, permutated_control_qubit);
                else
                  ::ket::mpi::gate::page::cphase_shift_coeff_tp(
                    mpi_policy, parallel_policy, local_state,
                    phase_coefficient, permutated_target_qubit, permutated_control_qubit,
                    communicator.rank(environment));
              }
              else if (::ket::mpi::page::is_on_page(permutated_control_qubit, local_state))
                ::ket::mpi::gate::page::cphase_shift_coeff_cp(
                  mpi_policy, parallel_policy, local_state,
                  phase_coefficient, permutated_target_qubit, permutated_control_qubit,
                  communicator.rank(environment));
              else
                ::ket::mpi::utility::diagonal_loop(
                  mpi_policy, parallel_policy,
                  local_state, permutation, communicator, environment, target_qubit,
                  [](auto const, StateInteger const) { },
                  [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; },
                  control_qubit);
              return true;
            }

            return false;
          }
# endif // KET_USE_DIAGONAL_LOOP

          // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::phase_shift_detail::maybe_apply_diagonal_phase_shift_coeff(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment,
                  phase_coefficient, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); }));

            return ::ket::mpi::gate::runtime::local::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase_coefficient, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::phase_shift_detail::maybe_apply_diagonal_phase_shift_coeff(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment,
                  phase_coefficient, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              control_qubits | boost::adaptors::transformed(
                [](control_qubit_type const control_qubit) { return control_qubit.qubit(); }));

            return ::ket::mpi::gate::runtime::local::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase_coefficient, control_qubits);
          }

          // Case 2: the first argument of qubits is ket::qubit<S, B>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::phase_shift_detail::maybe_apply_diagonal_phase_shift_coeff(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment,
                  phase_coefficient, target_qubit, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase_coefficient, target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
# ifdef KET_USE_DIAGONAL_LOOP
            if (::ket::mpi::gate::runtime::phase_shift_detail::maybe_apply_diagonal_phase_shift_coeff(
                  mpi_policy, parallel_policy, local_state, permutation, communicator, environment,
                  phase_coefficient, target_qubit, control_qubits))
              return local_state;
# endif // KET_USE_DIAGONAL_LOOP

            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase_coefficient, target_qubit, control_qubits);
          }
        } // namespace phase_shift_detail

        namespace ranges
        {
          // Case 1: the first argument of qubits is ket::control<ket::qubit<S, B>>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase(coeff) "),
                  phase_coefficient),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase(coeff) "),
                  phase_coefficient),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase_coefficient, control_qubits);
          }

          // Case 2: the first argument of qubits is ket::qubit<S, B>
          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase(coeff) "),
                  phase_coefficient),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient, target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitsRange>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase(coeff) "),
                  phase_coefficient),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase_coefficient, target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename Complex>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient, target_qubit, control_qubits);
          }

          template <
            typename MpiPolicy, typename ParallelPolicy,
            typename RandomAccessRange, typename StateInteger, typename BitInteger,
            typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
          inline auto phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase_coefficient, target_qubit, control_qubits);
          }
        } // namespace ranges

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitIterator>
        inline auto phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase_coefficient, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitIterator>
        inline auto phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase_coefficient, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitIterator>
        inline auto phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase_coefficient, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitIterator>
        inline auto phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase_coefficient, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename Complex>
        inline auto phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient, target_qubit);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
        inline auto phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase_coefficient, target_qubit);
        }

        namespace ranges
        {
          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitsRange>
          inline auto adj_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::conj;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase(coeff)) "),
                  phase_coefficient),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, conj(phase_coefficient), control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitsRange>
          inline auto adj_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::conj;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase(coeff)) "),
                  phase_coefficient),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, conj(phase_coefficient), control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitsRange>
          inline auto adj_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::conj;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase(coeff)) "),
                  phase_coefficient),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, conj(phase_coefficient), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitsRange>
          inline auto adj_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::conj;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase(coeff)) "),
                  phase_coefficient),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, conj(phase_coefficient), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex>
          inline auto adj_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using std::conj;
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase(coeff)) "}, phase_coefficient, ' ', target_qubit),
              environment};
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, conj(phase_coefficient), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
          inline auto adj_phase_shift_coeff(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Complex const& phase_coefficient,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using std::conj;
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase(coeff)) "}, phase_coefficient, ' ', target_qubit),
              environment};
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, conj(phase_coefficient), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase "),
                  phase),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              ::ket::utility::exp_i<complex_type>(phase), control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase "),
                  phase),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              ::ket::utility::exp_i<complex_type>(phase), control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase "),
                  phase),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase "),
                  phase),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
          inline auto phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase, ' ', target_qubit),
              environment};
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"Phase "}, phase, ' ', target_qubit),
              environment};
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              ::ket::utility::exp_i<complex_type>(phase), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase) "),
                  phase),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              ::ket::utility::exp_i<complex_type>(-phase), control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(std::max(num_control_qubits, std::size_t{1u}) - std::size_t{1u}, 'C').append("Phase) "),
                  phase),
                control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              ::ket::utility::exp_i<complex_type>(-phase), control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase) "),
                  phase),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              ::ket::utility::exp_i<complex_type>(-phase), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase) "),
                  phase),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              ::ket::utility::exp_i<complex_type>(-phase), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
          inline auto adj_phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase, ' ', target_qubit),
              environment};
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              ::ket::utility::exp_i<complex_type>(-phase), target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto adj_phase_shift(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using complex_type = ::ket::utility::meta::range_value_t<RandomAccessRange>;
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::utility::generate_logger_string(std::string{"Adj(Phase) "}, phase, ' ', target_qubit),
              environment};
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift_coeff(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              ::ket::utility::exp_i<complex_type>(-phase), target_qubit, control_qubits);
          }
        } // namespace ranges

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitIterator>
        inline auto adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase_coefficient, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitIterator>
        inline auto adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase_coefficient, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex, typename ControlQubitIterator>
        inline auto adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase_coefficient, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex, typename ControlQubitIterator>
        inline auto adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase_coefficient, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Complex>
        inline auto adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase_coefficient, target_qubit);
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Complex>
        inline auto adj_phase_shift_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Complex const& phase_coefficient, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift_coeff(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase_coefficient, target_qubit);
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase, target_qubit);
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase, target_qubit);
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase, target_qubit);
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto adj_phase_shift(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase, target_qubit);
        }

        namespace ranges
        {
          template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto phase_shift(
            ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::phase_shift(
              ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto phase_shift(
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::phase_shift(
              ::ket::mpi::utility::policy::make_simple_mpi(),
              ::ket::utility::policy::make_sequential(),
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto adj_phase_shift(
            ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
              ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }

          template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
          inline auto adj_phase_shift(
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            Args&&... args)
          -> RandomAccessRange&
          {
            return ::ket::mpi::gate::runtime::ranges::adj_phase_shift(
              ::ket::mpi::utility::policy::make_simple_mpi(),
              ::ket::utility::policy::make_sequential(),
              local_state, permutation, buffer, std::forward<Args>(args)...);
          }
        } // namespace ranges

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto phase_shift(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::phase_shift(
            ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto phase_shift(
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::phase_shift(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            ::ket::utility::policy::make_sequential(),
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto adj_phase_shift(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::adj_phase_shift(
            ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        template <typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename... Args>
        inline auto adj_phase_shift(
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          Args&&... args)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::adj_phase_shift(
            ::ket::mpi::utility::policy::make_simple_mpi(),
            ::ket::utility::policy::make_sequential(),
            local_state, permutation, buffer, std::forward<Args>(args)...);
        }

        namespace local
        {
          namespace dispatch
          {
            template <typename LocalState>
            struct transpage_phase_shift2
            {
              template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
                Real const phase1, Real const phase2,
                ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_phase_shift2<LocalState>

            template <typename LocalState>
            struct transpage_adj_phase_shift2
            {
              template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
                Real const phase1, Real const phase2,
                ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_adj_phase_shift2<LocalState>

            template <typename LocalState>
            struct transpage_phase_shift3
            {
              template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
                Real const phase1, Real const phase2, Real const phase3,
                ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_phase_shift3<LocalState>

            template <typename LocalState>
            struct transpage_adj_phase_shift3
            {
              template <typename ParallelPolicy, typename RandomAccessRange, typename Real, typename StateInteger, typename BitInteger, typename PermutatedControlQubitsRange>
              [[noreturn]] static auto call(
                ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
                Real const phase1, Real const phase2, Real const phase3,
                ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
                PermutatedControlQubitsRange const& permutated_control_qubits)
              -> RandomAccessRange&
              { throw 1; }
            }; // struct transpage_adj_phase_shift3<LocalState>
          } // namespace dispatch

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_target_qubit = permutation[target_qubit];
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment,
              boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
              permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(
                  local_state, boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
                  permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_phase_shift2<local_state_type>::call(
                parallel_policy, local_state, phase1, phase2, permutated_target_qubit, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, phase1, phase2, permutated_target_qubit, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::phase_shift2(
                  parallel_policy, first, last, phase1, phase2, permutated_target_qubit.qubit(),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_target_qubit = permutation[target_qubit];
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment,
              boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
              permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(
                  local_state, boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
                  permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_adj_phase_shift2<local_state_type>::call(
                parallel_policy, local_state, phase1, phase2, permutated_target_qubit, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, phase1, phase2, permutated_target_qubit, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::adj_phase_shift2(
                  parallel_policy, first, last, phase1, phase2, permutated_target_qubit.qubit(),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_target_qubit = permutation[target_qubit];
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment,
              boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
              permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(
                  local_state, boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
                  permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_phase_shift3<local_state_type>::call(
                parallel_policy, local_state, phase1, phase2, phase3, permutated_target_qubit, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, phase1, phase2, phase3, permutated_target_qubit, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::phase_shift3(
                  parallel_policy, first, last, phase1, phase2, phase3, permutated_target_qubit.qubit(),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            auto const permutated_target_qubit = permutation[target_qubit];
            auto const permutated_control_qubits
              = control_qubits | boost::adaptors::transformed(
                  [&permutation](control_qubit_type const control_qubit) { return permutation[control_qubit]; });

            ::ket::mpi::gate::detail::runtime::ranges::assert_all_qubits_are_local(
              mpi_policy, local_state, communicator, environment,
              boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
              permutated_control_qubits);

            if (::ket::mpi::page::runtime::ranges::any_on_page(
                  local_state, boost::make_iterator_range(&permutated_target_qubit, &permutated_target_qubit + 1),
                  permutated_control_qubits))
            {
              using local_state_type = std::remove_const_t<std::remove_reference_t<RandomAccessRange>>;
              return ::ket::mpi::gate::runtime::local::dispatch::transpage_adj_phase_shift3<local_state_type>::call(
                parallel_policy, local_state, phase1, phase2, phase3, permutated_target_qubit, permutated_control_qubits);
            }

            return ::ket::mpi::utility::for_each_local_range(
              mpi_policy, local_state, communicator, environment,
              [parallel_policy, phase1, phase2, phase3, permutated_target_qubit, permutated_control_qubits](auto const first, auto const last)
              {
                ::ket::gate::runtime::qubit_ranges::adj_phase_shift3(
                  parallel_policy, first, last, phase1, phase2, phase3, permutated_target_qubit.qubit(),
                  permutated_control_qubits | boost::adaptors::transformed(
                    [](typename ::ket::utility::meta::range_value_t<decltype(permutated_control_qubits)> const permutated_control_qubit)
                    { return permutated_control_qubit.qubit(); }));
              });
          }
        } // namespace local

        namespace phase_shift_detail
        {
          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::adj_phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::adj_phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::adj_phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::utility::meta::range_value_t<ControlQubitsRange>;
            ::ket::mpi::utility::runtime::ranges::maybe_interchange_qubits(
              mpi_policy, parallel_policy,
              local_state, permutation, buffer, datatype, communicator, environment,
              boost::join(
                boost::make_iterator_range(&target_qubit, &target_qubit + 1),
                control_qubits | boost::adaptors::transformed(
                  [](control_qubit_type const control_qubit) { return control_qubit.qubit(); })));

            return ::ket::mpi::gate::runtime::local::adj_phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }
        } // namespace phase_shift_detail

        namespace ranges
        {
          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase "), phase1, ' ', phase2),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase "), phase1, ' ', phase2),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
          inline auto phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase) "), phase1, ' ', phase2),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::adj_phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase) "), phase1, ' ', phase2),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::adj_phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
          inline auto adj_phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto adj_phase_shift2(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_phase_shift2(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase "), phase1, ' ', phase2, ' ', phase3),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string(num_control_qubits, 'C').append("Phase "), phase1, ' ', phase2, ' ', phase3),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
          inline auto phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase) "), phase1, ' ', phase2, ' ', phase3),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::adj_phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
              phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitsRange>
          inline auto adj_phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit, ControlQubitsRange const& control_qubits)
          -> RandomAccessRange&
          {
            using std::begin;
            using std::end;
            auto const num_control_qubits = static_cast<std::size_t>(std::distance(begin(control_qubits), end(control_qubits)));
            ::ket::mpi::utility::log_with_time_guard<char> print{
              ::ket::mpi::gate::detail::runtime::append_qubits_string(
                ::ket::mpi::utility::generate_logger_string(
                  std::string{"Adj("}.append(num_control_qubits, 'C').append("Phase) "), phase1, ' ', phase2, ' ', phase3),
                target_qubit, control_qubits),
              environment};

            return ::ket::mpi::gate::runtime::phase_shift_detail::adj_phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
              phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
          inline auto adj_phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }

          template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
          inline auto adj_phase_shift3(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            RandomAccessRange& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator, yampi::environment const& environment,
            Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
          -> RandomAccessRange&
          {
            using control_qubit_type = ::ket::control< ::ket::qubit<StateInteger, BitInteger> >;
            std::array<control_qubit_type, 0u> const control_qubits{};
            return ::ket::mpi::gate::runtime::ranges::adj_phase_shift3(
              mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, phase3, target_qubit, control_qubits);
          }
        } // namespace ranges

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift2(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase1, phase2, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift2(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase1, phase2, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::phase_shift2(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, target_qubit); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::phase_shift2(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, target_qubit); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift2(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase1, phase2, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift2(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase1, phase2, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto adj_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_phase_shift2(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, target_qubit); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto adj_phase_shift2(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_phase_shift2(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, target_qubit); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift3(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase1, phase2, phase3, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::phase_shift3(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase1, phase2, phase3, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::phase_shift3(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, phase3, target_qubit); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::phase_shift3(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, phase3, target_qubit); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift3(
            mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment,
            phase1, phase2, phase3, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real, typename ControlQubitIterator>
        inline auto adj_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit,
          ControlQubitIterator const control_qubit_first, ControlQubitIterator const control_qubit_last)
        -> RandomAccessRange&
        {
          return ::ket::mpi::gate::runtime::ranges::adj_phase_shift3(
            mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment,
            phase1, phase2, phase3, target_qubit, boost::make_iterator_range(control_qubit_first, control_qubit_last));
        }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename Real>
        inline auto adj_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_phase_shift3(mpi_policy, parallel_policy, local_state, permutation, buffer, communicator, environment, phase1, phase2, phase3, target_qubit); }

        template <typename MpiPolicy, typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator, typename DerivedDatatype, typename Real>
        inline auto adj_phase_shift3(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector< ::ket::utility::meta::range_value_t<RandomAccessRange>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment,
          Real const phase1, Real const phase2, Real const phase3, ::ket::qubit<StateInteger, BitInteger> const target_qubit)
        -> RandomAccessRange&
        { return ::ket::mpi::gate::runtime::ranges::adj_phase_shift3(mpi_policy, parallel_policy, local_state, permutation, buffer, datatype, communicator, environment, phase1, phase2, phase3, target_qubit); }
      } // namespace runtime
    } // namespace gate
  } // namespace mpi
} // namespace ket

#endif // KET_MPI_GATE_DETAIL_PHASE_SHIFT_RUNTIME_HPP
