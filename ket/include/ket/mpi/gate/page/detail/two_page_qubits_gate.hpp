#ifndef KET_MPI_GATE_PAGE_DETAIL_TWO_PAGE_QUBITS_GATE_HPP
# define KET_MPI_GATE_PAGE_DETAIL_TWO_PAGE_QUBITS_GATE_HPP

# include <cstddef>
# include <cassert>
# include <algorithm>
# include <iterator>

# include <boost/range/size.hpp>

# include <ket/meta/bit_integer_of.hpp>
# include <ket/meta/state_integer_of.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        namespace detail
        {
          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename RandomAccessRange, typename Qubit1, typename Qubit2, typename Function>
          [[noreturn]] inline RandomAccessRange& two_page_qubits_gate(
            ParallelPolicy const,
            RandomAccessRange& local_state,
            ::ket::mpi::permutated<Qubit1> const, ::ket::mpi::permutated<Qubit2> const,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"two_page_qubits_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename Qubit1, typename Qubit2, typename Function>
          [[noreturn]] inline ::ket::mpi::state<Complex, false, Allocator>& two_page_qubits_gate(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, false, Allocator>& local_state,
            ::ket::mpi::permutated<Qubit1> const, ::ket::mpi::permutated<Qubit2> const,
            Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"two_page_qubits_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename Qubit1, typename Qubit2, typename Function>
          inline ::ket::mpi::state<Complex, true, Allocator>& two_page_qubits_gate(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, true, Allocator>& local_state,
            ::ket::mpi::permutated<Qubit1> const permutated_qubit1,
            ::ket::mpi::permutated<Qubit2> const permutated_qubit2,
            Function&& function)
          {
            using bit_integer_type = typename ::ket::meta::bit_integer_of<Qubit1>::type;
            static_assert(
              std::is_same<bit_integer_type, typename ::ket::meta::bit_integer_of<Qubit2>::type>::value,
              "Qubit1 and Qubit2 should have the same BitInteger type");
            using state_integer_type = typename ::ket::meta::state_integer_of<Qubit1>::type;
            static_assert(
              std::is_same<state_integer_type, typename ::ket::meta::state_integer_of<Qubit2>::type>::value,
              "Qubit1 and Qubit2 should have the same StateInteger type");
            assert(local_state.num_page_qubits() >= std::size_t{2u});
            assert(::ket::mpi::page::is_on_page(permutated_qubit1, local_state));
            assert(::ket::mpi::page::is_on_page(permutated_qubit2, local_state));

            auto const num_nonpage_local_qubits
              = static_cast<bit_integer_type>(local_state.num_local_qubits() - local_state.num_page_qubits());

            using permutated_qubit_type = decltype(::ket::mpi::remove_control(permutated_qubit1));
            static_assert(
              std::is_same<permutated_qubit_type, decltype(::ket::mpi::remove_control(permutated_qubit2))>::value,
              "Qubit1 and Qubit2 should become the same after removing ket::control");
            auto const minmax_permutated_qubits
              = static_cast<std::pair<permutated_qubit_type, permutated_qubit_type>>(
                  std::minmax(::ket::mpi::remove_control(permutated_qubit1), ::ket::mpi::remove_control(permutated_qubit2)));
            auto const permutated_qubit1_mask
              = ::ket::utility::integer_exp2<state_integer_type>(
                  permutated_qubit1 - static_cast<bit_integer_type>(num_nonpage_local_qubits));
            auto const permutated_qubit2_mask
              = ::ket::utility::integer_exp2<state_integer_type>(
                  permutated_qubit2 - static_cast<bit_integer_type>(num_nonpage_local_qubits));
            auto const lower_bits_mask
              = ::ket::utility::integer_exp2<state_integer_type>(
                  minmax_permutated_qubits.first - static_cast<bit_integer_type>(num_nonpage_local_qubits)) - state_integer_type{1u};
            auto const middle_bits_mask
              = (::ket::utility::integer_exp2<state_integer_type>(
                   minmax_permutated_qubits.second - (bit_integer_type{1u} + num_nonpage_local_qubits)) - state_integer_type{1u})
                xor lower_bits_mask;
            auto const upper_bits_mask = compl (lower_bits_mask bitor middle_bits_mask);

            auto const num_pages = local_state.num_pages();
            auto const num_data_blocks = local_state.num_data_blocks();
            for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
              for (auto page_index_wo_qubits = std::size_t{0u};
                   page_index_wo_qubits < num_pages / 4u; ++page_index_wo_qubits)
              {
                // x0_2x0_1x
                auto const page_index_00
                  = ((page_index_wo_qubits bitand upper_bits_mask) << 2u)
                    bitor ((page_index_wo_qubits bitand middle_bits_mask) << 1u)
                    bitor (page_index_wo_qubits bitand lower_bits_mask);
                // x0_2x1_1x
                auto const page_index_01 = page_index_00 bitor permutated_qubit1_mask;
                // x1_2x0_1x
                auto const page_index_10 = page_index_00 bitor permutated_qubit2_mask;
                // x1_2x1_1x
                auto const page_index_11 = page_index_10 bitor permutated_qubit1_mask;

                auto const page_range_00 = local_state.page_range(std::make_pair(data_block_index, page_index_00));
                auto const first_00 = std::begin(page_range_00);
                auto const page_range_01 = local_state.page_range(std::make_pair(data_block_index, page_index_01));
                auto const first_01 = std::begin(page_range_01);
                auto const page_range_10 = local_state.page_range(std::make_pair(data_block_index, page_index_10));
                auto const first_10 = std::begin(page_range_10);
                auto const page_range_11 = local_state.page_range(std::make_pair(data_block_index, page_index_11));
                auto const first_11 = std::begin(page_range_11);

                using ::ket::utility::loop_n;
                loop_n(
                  parallel_policy,
                  boost::size(page_range_11) >> num_operated_nonpage_qubits,
                  [first_00, first_01, first_10, first_11, &function](state_integer_type const index_wo_nonpage_qubits, int const thread_index)
                  { function(first_00, first_01, first_10, first_11, index_wo_nonpage_qubits, thread_index); });
              }

            return local_state;
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_TWO_PAGE_QUBIT_GATES_HPP
