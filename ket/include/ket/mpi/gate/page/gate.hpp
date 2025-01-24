#ifndef KET_MPI_GATE_PAGE_GATE_HPP
# define KET_MPI_GATE_PAGE_GATE_HPP

# include <cassert>
# include <algorithm>
# include <iterator>
# include <vector>
# include <array>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/gate/gate.hpp>
# include <ket/gate/utility/index_with_qubits.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/variadic/transform.hpp>
# include <ket/utility/variadic/all_of.hpp>
# include <ket/meta/bit_integer_of.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/page/any_on_page.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>


# ifdef KET_USE_ON_CACHE_STATE_VECTOR
namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename ContiguousIterator, typename StateInteger,
          typename Function, typename Qubit, typename... Qubits>
        [[noreturn]] inline auto gate(
          ParallelPolicy const,
          RandomAccessRange&, ContiguousIterator const, ContiguousIterator const, StateInteger const,
          Function&&, ::ket::mpi::permutated<Qubit> const, ::ket::mpi::permutated<Qubits> const...)
        -> RandomAccessRange&
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"gate"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ContiguousIterator, typename StateInteger,
          typename Function, typename Qubit, typename... Qubits>
        [[noreturn]] inline auto gate(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator>&, ContiguousIterator const, ContiguousIterator const, StateInteger const,
          Function&&, ::ket::mpi::permutated<Qubit> const, ::ket::mpi::permutated<Qubits> const...)
        -> ::ket::mpi::state<Complex, false, Allocator>&
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"gate"}; }

        // Case 2) Some operated qubits are page qubits
        // Note that num. page qubits are smaller than or equal to num. off-cache qubits
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        namespace impl
        {
          template <
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename ContiguousIterator, typename StateInteger,
            typename Function, typename Qubit, typename... Qubits, std::size_t... indices_for_permutated_qubits>
          inline auto gate(
            std::index_sequence<indices_for_permutated_qubits...> const,
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            ContiguousIterator const on_cache_state_first, ContiguousIterator const on_cache_state_last,
            StateInteger const data_block_index,
            Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
          -> ::ket::mpi::state<Complex, true, Allocator>&
          {
            static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
#   if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
                [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
                Qubit{}, Qubits{}...),
              "state_integer_type's of Qubit and Qubits should be the same as StateInteger");
#   else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
                Qubit{}, Qubits{}...),
              "state_integer_type's of Qubit and Qubits should be the same as StateInteger");
#   endif // __cpp_constexpr >= 201603L

            using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;
            static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
#   if __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
                [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
                Qubits{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#   else // __cpp_constexpr >= 201603L
            static_assert(
              ::ket::utility::variadic::proj::all_of(
                ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
                Qubits{}...),
              "bit_integer_type's of Qubit and Qubits should be the same");
#   endif // __cpp_constexpr >= 201603L

            // Case 1) should be resolved before calling this function
            assert(::ket::mpi::page::any_on_page(local_state, permutated_qubit, permutated_qubits...));

            assert(local_state.num_local_qubits() > local_state.num_page_qubits());
            auto const on_cache_state_size = static_cast<StateInteger>(on_cache_state_last - on_cache_state_first);
            auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(on_cache_state_size);
            assert(::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits) == on_cache_state_size);
#   ifndef NDEBUG
            auto const num_nonpage_qubits = static_cast<bit_integer_type>(local_state.num_local_qubits() - local_state.num_page_qubits());
#   endif // NDEBUG
            assert(num_nonpage_qubits >= num_on_cache_qubits);
            assert(static_cast<bit_integer_type>(local_state.num_local_qubits()) > num_on_cache_qubits); // because num_page_qubits >= 1
            auto const num_off_cache_qubits = static_cast<bit_integer_type>(local_state.num_local_qubits()) - num_on_cache_qubits;
            assert(static_cast<bit_integer_type>(local_state.num_page_qubits()) <= num_off_cache_qubits);

            constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
            assert(num_operated_qubits < num_on_cache_qubits);

            // ppxx|yyyy|zzzzzz: local qubits
            // * ppxx: off-cache qubits
            // * yyyy|zzzzzz: on-cache qubits
            //   - yyyy: chunk qubits
            // * ppxx|yyyy: tag qubits
            // * zzzzzz: nontag qubits

            using qubit_type = ::ket::qubit<StateInteger, bit_integer_type>;
            using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
            auto const least_significant_off_cache_permutated_qubit = permutated_qubit_type{num_on_cache_qubits};

            // operated_on_cache_permutated_qubits_first, operated_on_cache_permutated_qubits_last
            std::array<permutated_qubit_type, num_operated_qubits> sorted_permutated_qubits{::ket::mpi::remove_control(permutated_qubit), ::ket::mpi::remove_control(permutated_qubits)...};
            using std::begin;
            using std::end;
            std::sort(begin(sorted_permutated_qubits), end(sorted_permutated_qubits));
            auto const operated_on_cache_permutated_qubits_last
              = std::lower_bound(begin(sorted_permutated_qubits), end(sorted_permutated_qubits), least_significant_off_cache_permutated_qubit);
            auto const operated_on_cache_permutated_qubits_first = begin(sorted_permutated_qubits);
            auto const operated_off_cache_permutated_qubits_first = operated_on_cache_permutated_qubits_last;
            auto const operated_off_cache_permutated_qubits_last = end(sorted_permutated_qubits);
            // num_page_qubits <= num_off_cache_qubits and there are some operated page qubits -> there are some operated off-cache qubits
            assert(operated_off_cache_permutated_qubits_first != operated_off_cache_permutated_qubits_last);

            // Case 2-1) There is no operated on-cache qubit
            //   ex1: ppxx|yyy|zzzzzzz
            //        ^^ ^             <- operated qubits
            //   ex2: pppp|yyy|zzzzzzz
            //        ^^ ^             <- operated qubits
            if (operated_on_cache_permutated_qubits_first == operated_on_cache_permutated_qubits_last)
            {
              // num_chunk_qubits, chunk_size, least_significant_chunk_permutated_qubit, num_tag_qubits, num_nontag_qubits
              constexpr auto num_chunk_qubits = num_operated_qubits;
              constexpr auto num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<StateInteger>(num_chunk_qubits);
              auto const chunk_size = on_cache_state_size / num_chunks_in_on_cache_state;
              auto const least_significant_chunk_permutated_qubit = least_significant_off_cache_permutated_qubit - num_chunk_qubits;
              auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
              auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;
              auto const num_nonpage_tag_qubits = num_tag_qubits - static_cast<bit_integer_type>(local_state.num_page_qubits());

              // unsorted_tag_qubits, sorted_tag_qubits_with_sentinel
              std::array<qubit_type, num_operated_qubits> unsorted_tag_qubits{
                (::ket::mpi::remove_control(permutated_qubit) - num_nontag_qubits).qubit(),
                (::ket::mpi::remove_control(permutated_qubits) - num_nontag_qubits).qubit()...};
              std::array<qubit_type, num_operated_qubits + bit_integer_type{1u}> sorted_tag_qubits_with_sentinel{
                (::ket::mpi::remove_control(permutated_qubit) - num_nontag_qubits).qubit(),
                (::ket::mpi::remove_control(permutated_qubits) - num_nontag_qubits).qubit()...,
                qubit_type{num_tag_qubits}};
              std::sort(begin(sorted_tag_qubits_with_sentinel), std::prev(end(sorted_tag_qubits_with_sentinel)));

              auto const tag_loop_size = ::ket::utility::integer_exp2<StateInteger>(num_tag_qubits - num_operated_qubits);
              for (auto tag_index_wo_qubits = StateInteger{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
              {
                for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                {
                  auto const tag_index = ::ket::gate::utility::index_with_qubits(tag_index_wo_qubits, chunk_index, unsorted_tag_qubits, sorted_tag_qubits_with_sentinel);
                  auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
                  auto const page_index = tag_index >> num_nonpage_tag_qubits;
                  auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
                  ::ket::utility::copy_n(
                    parallel_policy,
                    page_first + nonpage_tag_index * chunk_size, chunk_size, on_cache_state_first + chunk_index * chunk_size);
                }

                ::ket::gate::nocache::gate(
                  parallel_policy, on_cache_state_first, on_cache_state_last, std::forward<Function>(function),
                  least_significant_chunk_permutated_qubit.qubit(),
                  (least_significant_chunk_permutated_qubit + bit_integer_type{1u} + bit_integer_type{indices_for_permutated_qubits}).qubit()...);

                for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
                {
                  auto const tag_index = ::ket::gate::utility::index_with_qubits(tag_index_wo_qubits, chunk_index, unsorted_tag_qubits, sorted_tag_qubits_with_sentinel);
                  auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
                  auto const page_index = tag_index >> num_nonpage_tag_qubits;
                  auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
                  ::ket::utility::copy_n(
                    parallel_policy,
                    on_cache_state_first + chunk_index * chunk_size, chunk_size, page_first + nonpage_tag_index * chunk_size);
                }
              }

              return local_state;
            }

            // Case 2-2) There are some operated on-cache qubits
            //   ex: ppxx|yyy|zzzzzzz
            //        ^^   ^    ^     <- operated qubits
            // least_significant_chunk_permutated_qubit, num_chunk_qubits, chunk_size, num_tag_qubits, num_nontag_qubits
            assert(operated_on_cache_permutated_qubits_first != operated_on_cache_permutated_qubits_last);
            auto operated_on_cache_permutated_qubits_iter = std::prev(operated_on_cache_permutated_qubits_last);
            auto free_most_significant_on_cache_permutated_qubit = least_significant_off_cache_permutated_qubit - bit_integer_type{1u};
            auto const num_operated_off_cache_qubits
              = static_cast<bit_integer_type>(operated_off_cache_permutated_qubits_last - operated_off_cache_permutated_qubits_first);
            for (auto num_found_operated_off_cache_qubits = bit_integer_type{0u};
                 num_found_operated_off_cache_qubits < num_operated_off_cache_qubits; ++num_found_operated_off_cache_qubits)
              while (free_most_significant_on_cache_permutated_qubit-- == *operated_on_cache_permutated_qubits_iter)
                if (operated_on_cache_permutated_qubits_iter != operated_on_cache_permutated_qubits_first)
                  --operated_on_cache_permutated_qubits_iter;
            auto const least_significant_chunk_permutated_qubit = free_most_significant_on_cache_permutated_qubit + bit_integer_type{1u};
            auto const num_chunk_qubits = static_cast<bit_integer_type>(least_significant_off_cache_permutated_qubit - least_significant_chunk_permutated_qubit);
            assert(num_chunk_qubits <= num_operated_qubits);
            auto const num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<StateInteger>(num_chunk_qubits);
            auto const chunk_size = on_cache_state_size / num_chunks_in_on_cache_state;
            auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
            auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;
            auto const num_nonpage_tag_qubits = num_tag_qubits - static_cast<bit_integer_type>(local_state.num_page_qubits());

            // unsorted_tag_qubits, modified_operated_qubits
            auto unsorted_tag_qubits = std::vector<qubit_type>{};
            unsorted_tag_qubits.reserve(num_chunk_qubits);
            auto present_chunk_permutated_qubit = least_significant_chunk_permutated_qubit;
            auto const modified_operated_qubits
              = ::ket::utility::variadic::transform(
                  [least_significant_chunk_permutated_qubit, num_nontag_qubits, &unsorted_tag_qubits, &present_chunk_permutated_qubit](auto permutated_qubit)
                  {
                    if (permutated_qubit < least_significant_chunk_permutated_qubit)
                      return permutated_qubit.qubit();

                    unsorted_tag_qubits.push_back(::ket::remove_control(permutated_qubit.qubit()) - num_nontag_qubits);
                    return static_cast<decltype(permutated_qubit.qubit())>((present_chunk_permutated_qubit++).qubit());
                  },
                  permutated_qubit, permutated_qubits...);
            assert(present_chunk_permutated_qubit == least_significant_off_cache_permutated_qubit);
            assert(static_cast<bit_integer_type>(unsorted_tag_qubits.size()) == num_chunk_qubits);

            // sorted_tag_qubits_with_sentinel
            auto sorted_tag_qubits_with_sentinel = std::vector<qubit_type>{};
            sorted_tag_qubits_with_sentinel.reserve(unsorted_tag_qubits.size() + 1u);
            std::copy(begin(unsorted_tag_qubits), end(unsorted_tag_qubits), std::back_inserter(sorted_tag_qubits_with_sentinel));
            sorted_tag_qubits_with_sentinel.push_back(qubit_type{num_tag_qubits});
            std::sort(begin(sorted_tag_qubits_with_sentinel), std::prev(end(sorted_tag_qubits_with_sentinel)));
            assert(sorted_tag_qubits_with_sentinel.size() == unsorted_tag_qubits.size() + 1u);

            auto const tag_loop_size = ::ket::utility::integer_exp2<StateInteger>(num_tag_qubits - num_chunk_qubits); // num_chunk_qubits == operated_tag_qubits.size()
            for (auto tag_index_wo_qubits = StateInteger{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
            {
              for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
              {
                auto const tag_index
                  = ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, chunk_index,
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel));
                auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
                auto const page_index = tag_index >> num_nonpage_tag_qubits;
                auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
                ::ket::utility::copy_n(
                  parallel_policy,
                  page_first + nonpage_tag_index * chunk_size, chunk_size, on_cache_state_first + chunk_index * chunk_size);
              }

              ::ket::gate::nocache::gate(
                parallel_policy, on_cache_state_first, on_cache_state_last, std::forward<Function>(function),
                std::get<0u>(modified_operated_qubits), std::get<1u + indices_for_permutated_qubits>(modified_operated_qubits)...);

              for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
              {
                auto const tag_index
                  = ::ket::gate::utility::index_with_qubits(
                      tag_index_wo_qubits, chunk_index,
                      begin(unsorted_tag_qubits), end(unsorted_tag_qubits),
                      begin(sorted_tag_qubits_with_sentinel), end(sorted_tag_qubits_with_sentinel));
                auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
                auto const page_index = tag_index >> num_nonpage_tag_qubits;
                auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
                ::ket::utility::copy_n(
                  parallel_policy,
                  on_cache_state_first + chunk_index * chunk_size, chunk_size, page_first + nonpage_tag_index * chunk_size);
              }
            }

            return local_state;
          }
        } // namespace impl

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ContiguousIterator, typename StateInteger,
          typename Function, typename Qubit, typename... Qubits, typename IndicesForPermutatedQubits = std::make_index_sequence<sizeof...(Qubits)>>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ContiguousIterator const on_cache_state_first, ContiguousIterator const on_cache_state_last,
          StateInteger const data_block_index,
          Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> ::ket::mpi::state<Complex, true, Allocator>&
        {
          return ::ket::mpi::gate::page::impl::gate(
            IndicesForPermutatedQubits{},
            parallel_policy,
            local_state, on_cache_state_first, on_cache_state_last, data_block_index,
            std::forward<Function>(function), permutated_qubit, permutated_qubits...);
        }
# else // KET_USE_BIT_MASKS_EXPLICITLY
        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename ContiguousIterator, typename StateInteger,
          typename Function, typename Qubit, typename... Qubits>
        inline auto gate(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ContiguousIterator const on_cache_state_first, ContiguousIterator const on_cache_state_last,
          StateInteger const data_block_index,
          Function&& function, ::ket::mpi::permutated<Qubit> const permutated_qubit, ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> ::ket::mpi::state<Complex, true, Allocator>&
        {
          static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
#   if __cpp_constexpr >= 201603L
          static_assert(
            ::ket::utility::variadic::proj::all_of(
              [](auto integer) { return std::is_same<decltype(integer), StateInteger>::value; },
              [](auto qubit) { return ::ket::meta::state_integer_t<decltype(qubit)>{}; },
              Qubit{}, Qubits{}...),
            "state_integer_type's of Qubit and Qubits should be the same as StateInteger");
#   else // __cpp_constexpr >= 201603L
          static_assert(
            ::ket::utility::variadic::proj::all_of(
              ::ket::gate::gate_detail::is_same_to<StateInteger>{}, ::ket::gate::gate_detail::state_integer_of{},
              Qubit{}, Qubits{}...),
            "state_integer_type's of Qubit and Qubits should be the same as StateInteger");
#   endif // __cpp_constexpr >= 201603L

          using bit_integer_type = ::ket::meta::bit_integer_t<Qubit>;
          static_assert(std::is_unsigned<bit_integer_type>::value, "bit_integer_type of Qubit should be unsigned");
#   if __cpp_constexpr >= 201603L
          static_assert(
            ::ket::utility::variadic::proj::all_of(
              [](auto integer) { return std::is_same<decltype(integer), bit_integer_type>::value; },
              [](auto qubit) { return ::ket::meta::bit_integer_t<decltype(qubit)>{}; },
              Qubits{}...),
            "bit_integer_type's of Qubit and Qubits should be the same");
#   else // __cpp_constexpr >= 201603L
          static_assert(
            ::ket::utility::variadic::proj::all_of(
              ::ket::gate::gate_detail::is_same_to<bit_integer_type>{}, ::ket::gate::gate_detail::bit_integer_of{},
              Qubits{}...),
            "bit_integer_type's of Qubit and Qubits should be the same");
#   endif // __cpp_constexpr >= 201603L

          // Case 1) should be resolved before calling this function
          assert(::ket::mpi::page::any_on_page(local_state, permutated_qubit, permutated_qubits...));

          assert(local_state.num_local_qubits() > local_state.num_page_qubits());
          auto const on_cache_state_size = static_cast<StateInteger>(on_cache_state_last - on_cache_state_first);
          auto const num_on_cache_qubits = ::ket::utility::integer_log2<bit_integer_type>(on_cache_state_size);
          assert(::ket::utility::integer_exp2<StateInteger>(num_on_cache_qubits) == on_cache_state_size);
#   ifndef NDEBUG
          auto const num_nonpage_qubits = static_cast<bit_integer_type>(local_state.num_local_qubits() - local_state.num_page_qubits());
#   endif // NDEBUG
          assert(num_nonpage_qubits >= num_on_cache_qubits);
          assert(static_cast<bit_integer_type>(local_state.num_local_qubits()) > num_on_cache_qubits); // because num_page_qubits >= 1
          auto const num_off_cache_qubits = static_cast<bit_integer_type>(local_state.num_local_qubits()) - num_on_cache_qubits;
          assert(static_cast<bit_integer_type>(local_state.num_page_qubits()) <= num_off_cache_qubits);

          constexpr auto num_operated_qubits = bit_integer_type{sizeof...(Qubits) + 1u};
          assert(num_operated_qubits < num_on_cache_qubits);

          // ppxx|yyyy|zzzzzz: local qubits
          // * ppxx: off-cache qubits
          // * yyyy|zzzzzz: on-cache qubits
          //   - yyyy: chunk qubits
          // * ppxx|yyyy: tag qubits
          // * zzzzzz: nontag qubits

          using qubit_type = ::ket::qubit<StateInteger, bit_integer_type>;
          using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
          auto const least_significant_off_cache_permutated_qubit = permutated_qubit_type{num_on_cache_qubits};

          // operated_on_cache_permutated_qubits_first, operated_on_cache_permutated_qubits_last
          std::array<permutated_qubit_type, num_operated_qubits> sorted_permutated_qubits{::ket::mpi::remove_control(permutated_qubit), ::ket::mpi::remove_control(permutated_qubits)...};
          using std::begin;
          using std::end;
          std::sort(begin(sorted_permutated_qubits), end(sorted_permutated_qubits));
          auto const operated_on_cache_permutated_qubits_last
            = std::lower_bound(begin(sorted_permutated_qubits), end(sorted_permutated_qubits), least_significant_off_cache_permutated_qubit);
          auto const operated_on_cache_permutated_qubits_first = begin(sorted_permutated_qubits);
          auto const operated_off_cache_permutated_qubits_first = operated_on_cache_permutated_qubits_last;
          auto const operated_off_cache_permutated_qubits_last = end(sorted_permutated_qubits);
          // num_page_qubits <= num_off_cache_qubits and there are some operated page qubits -> there are some operated off-cache qubits
          assert(operated_off_cache_permutated_qubits_first != operated_off_cache_permutated_qubits_last);

          // Case 2-1) There is no operated on-cache qubit
          //   ex1: ppxx|yyy|zzzzzzz
          //        ^^ ^             <- operated qubits
          //   ex2: pppp|yyy|zzzzzzz
          //        ^^ ^             <- operated qubits
          if (operated_on_cache_permutated_qubits_first == operated_on_cache_permutated_qubits_last)
          {
            // num_chunk_qubits, chunk_size, least_significant_chunk_permutated_qubit, num_tag_qubits, num_nontag_qubits
            constexpr auto num_chunk_qubits = num_operated_qubits;
            constexpr auto num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<StateInteger>(num_chunk_qubits);
            auto const chunk_size = on_cache_state_size / num_chunks_in_on_cache_state;
            auto const least_significant_chunk_permutated_qubit = least_significant_off_cache_permutated_qubit - num_chunk_qubits;
            auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
            auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;
            auto const num_nonpage_tag_qubits = num_tag_qubits - static_cast<bit_integer_type>(local_state.num_page_qubits());

            // on_cache_qubit_masks, on_cache_index_masks
            std::array<StateInteger, num_operated_qubits> on_cache_qubit_masks;
            for (auto index = bit_integer_type{0u}; index < num_operated_qubits; ++index)
              on_cache_qubit_masks[index] = StateInteger{1u} << (least_significant_chunk_permutated_qubit + index);
            std::array<StateInteger, num_operated_qubits + bit_integer_type{1u}> on_cache_index_masks;
            on_cache_index_masks.front() = (StateInteger{1u} << least_significant_chunk_permutated_qubit) - StateInteger{1u};
            // on_cache_index_masks.size() >= 2 => std::prev(end(on_cache_index_masks)) >= std::next(begin(on_cache_index_masks))
            std::fill(std::next(begin(on_cache_index_masks)), std::prev(end(on_cache_index_masks)), StateInteger{0u});
            on_cache_index_masks.back() = compl on_cache_index_masks.front();

            // tag_qubit_masks, tag_index_masks
            std::array<StateInteger, num_operated_qubits> tag_qubit_masks;
            ::ket::gate::gate_detail::make_qubit_masks(tag_qubit_masks, (permutated_qubit - num_nontag_qubits).qubit(), (permutated_qubits - num_nontag_qubits).qubit()...);
            std::array<StateInteger, num_operated_qubits + bit_integer_type{1u}> tag_index_masks;
            ::ket::gate::gate_detail::make_index_masks(tag_index_masks, (permutated_qubit - num_nontag_qubits).qubit(), (permutated_qubits - num_nontag_qubits).qubit()...);

            auto const tag_loop_size = ::ket::utility::integer_exp2<StateInteger>(num_tag_qubits - num_operated_qubits);
            for (auto tag_index_wo_qubits = StateInteger{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
            {
              for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
              {
                auto const tag_index = ::ket::gate::utility::index_with_qubits(tag_index_wo_qubits, chunk_index, tag_qubit_masks, tag_index_masks);
                auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
                auto const page_index = tag_index >> num_nonpage_tag_qubits;
                auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
                ::ket::utility::copy_n(
                  parallel_policy,
                  page_first + nonpage_tag_index * chunk_size, chunk_size, on_cache_state_first + chunk_index * chunk_size);
              }

              ::ket::gate::gate_detail::gate(parallel_policy, on_cache_state_first, on_cache_state_last, on_cache_qubit_masks, on_cache_index_masks, std::forward<Function>(function));

              for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
              {
                auto const tag_index = ::ket::gate::utility::index_with_qubits(tag_index_wo_qubits, chunk_index, tag_qubit_masks, tag_index_masks);
                auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
                auto const page_index = tag_index >> num_nonpage_tag_qubits;
                auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
                ::ket::utility::copy_n(
                  parallel_policy,
                  on_cache_state_first + chunk_index * chunk_size, chunk_size, page_first + nonpage_tag_index * chunk_size);
              }
            }

            return local_state;
          }

          // Case 2-2) There are some operated on-cache qubits
          //   ex: ppxx|yyy|zzzzzzz
          //        ^^   ^    ^     <- operated qubits
          // least_significant_chunk_permutated_qubit, num_chunk_qubits, chunk_size, num_tag_qubits, num_nontag_qubits
          assert(operated_on_cache_permutated_qubits_first != operated_on_cache_permutated_qubits_last);
          auto operated_on_cache_permutated_qubits_iter = std::prev(operated_on_cache_permutated_qubits_last);
          auto free_most_significant_on_cache_permutated_qubit = least_significant_off_cache_permutated_qubit - bit_integer_type{1u};
          auto const num_operated_off_cache_qubits
            = static_cast<bit_integer_type>(operated_off_cache_permutated_qubits_last - operated_off_cache_permutated_qubits_first);
          for (auto num_found_operated_off_cache_qubits = bit_integer_type{0u};
               num_found_operated_off_cache_qubits < num_operated_off_cache_qubits; ++num_found_operated_off_cache_qubits)
            while (free_most_significant_on_cache_permutated_qubit-- == *operated_on_cache_permutated_qubits_iter)
              if (operated_on_cache_permutated_qubits_iter != operated_on_cache_permutated_qubits_first)
                --operated_on_cache_permutated_qubits_iter;
          auto const least_significant_chunk_permutated_qubit = free_most_significant_on_cache_permutated_qubit + bit_integer_type{1u};
          auto const num_chunk_qubits = static_cast<bit_integer_type>(least_significant_off_cache_permutated_qubit - least_significant_chunk_permutated_qubit);
          assert(num_chunk_qubits <= num_operated_qubits);
          auto const num_chunks_in_on_cache_state = ::ket::utility::integer_exp2<StateInteger>(num_chunk_qubits);
          auto const chunk_size = on_cache_state_size / num_chunks_in_on_cache_state;
          auto const num_tag_qubits = num_off_cache_qubits + num_chunk_qubits;
          auto const num_nontag_qubits = num_on_cache_qubits - num_chunk_qubits;
          auto const num_nonpage_tag_qubits = num_tag_qubits - static_cast<bit_integer_type>(local_state.num_page_qubits());

          // operated_tag_qubits, on_cache_qubit_masks, on_cache_index_masks
          auto operated_tag_qubits = std::vector<qubit_type>{};
          operated_tag_qubits.reserve(num_chunk_qubits);
          auto present_chunk_permutated_qubit = least_significant_chunk_permutated_qubit;
          auto const modified_operated_qubits
            = ::ket::utility::variadic::transform(
                [least_significant_chunk_permutated_qubit, num_nontag_qubits, &operated_tag_qubits, &present_chunk_permutated_qubit](auto permutated_qubit)
                {
                  if (permutated_qubit < least_significant_chunk_permutated_qubit)
                    return permutated_qubit.qubit();

                  operated_tag_qubits.push_back(::ket::remove_control(permutated_qubit.qubit()) - num_nontag_qubits);
                  return static_cast<decltype(permutated_qubit.qubit())>((present_chunk_permutated_qubit++).qubit());
                },
                permutated_qubit, permutated_qubits...);
          assert(present_chunk_permutated_qubit == least_significant_off_cache_permutated_qubit);
          assert(static_cast<bit_integer_type>(operated_tag_qubits.size()) == num_chunk_qubits);
          std::array<StateInteger, num_operated_qubits> on_cache_qubit_masks{};
          ::ket::gate::gate_detail::make_qubit_masks(modified_operated_qubits, on_cache_qubit_masks);
          std::array<StateInteger, num_operated_qubits + 1u> on_cache_index_masks{};
          ::ket::gate::gate_detail::make_index_masks(modified_operated_qubits, on_cache_index_masks);

          // tag_qubit_masks, tag_index_masks
          auto tag_qubit_masks = std::vector<StateInteger>{};
          tag_qubit_masks.reserve(operated_tag_qubits.size());
          ::ket::gate::gate_detail::runtime::ranges::make_qubit_masks(operated_tag_qubits, std::back_inserter(tag_qubit_masks));
          assert(tag_qubit_masks.size() == operated_tag_qubits.size());
          auto tag_index_masks = std::vector<StateInteger>{};
          tag_index_masks.reserve(operated_tag_qubits.size() + 1u);
          ::ket::gate::gate_detail::runtime::ranges::make_index_masks(operated_tag_qubits, std::back_inserter(tag_index_masks));
          assert(tag_index_masks.size() == operated_tag_qubits.size() + 1u);

          auto const tag_loop_size = ::ket::utility::integer_exp2<StateInteger>(num_tag_qubits - num_chunk_qubits); // num_chunk_qubits == operated_tag_qubits.size()
          for (auto tag_index_wo_qubits = StateInteger{0u}; tag_index_wo_qubits < tag_loop_size; ++tag_index_wo_qubits)
          {
            for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
            {
              auto const tag_index = ::ket::gate::utility::index_with_qubits(tag_index_wo_qubits, chunk_index, begin(tag_qubit_masks), end(tag_qubit_masks), begin(tag_index_masks), end(tag_index_masks));
              auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
              auto const page_index = tag_index >> num_nonpage_tag_qubits;
              auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
              ::ket::utility::copy_n(
                parallel_policy,
                page_first + nonpage_tag_index * chunk_size, chunk_size, on_cache_state_first + chunk_index * chunk_size);
            }

            ::ket::gate::gate_detail::gate(parallel_policy, on_cache_state_first, on_cache_state_last, on_cache_qubit_masks, on_cache_index_masks, std::forward<Function>(function));

            for (auto chunk_index = StateInteger{0u}; chunk_index < num_chunks_in_on_cache_state; ++chunk_index)
            {
              auto const tag_index = ::ket::gate::utility::index_with_qubits(tag_index_wo_qubits, chunk_index, begin(tag_qubit_masks), end(tag_qubit_masks), begin(tag_index_masks), end(tag_index_masks));
              auto const nonpage_tag_index = tag_index bitand ((StateInteger{1u} << num_nonpage_tag_qubits) - StateInteger{1u});
              auto const page_index = tag_index >> num_nonpage_tag_qubits;
              auto const page_first = begin(local_state.page_range(std::make_pair(data_block_index, page_index)));
              ::ket::utility::copy_n(
                parallel_policy,
                on_cache_state_first + chunk_index * chunk_size, chunk_size, page_first + nonpage_tag_index * chunk_size);
            }
          }

          return local_state;
        }
# endif // KET_USE_BIT_MASKS_EXPLICITLY
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket
# endif // KET_USE_ON_CACHE_STATE_VECTOR


#endif // KET_MPI_GATE_PAGE_GATE_HPP
