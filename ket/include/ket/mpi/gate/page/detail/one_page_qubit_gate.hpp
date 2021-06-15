#ifndef KET_MPI_GATE_PAGE_DETAIL_ONE_PAGE_QUBIT_GATE_HPP
# define KET_MPI_GATE_PAGE_DETAIL_ONE_PAGE_QUBIT_GATE_HPP

# include <cstddef>
# include <cassert>
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
            typename RandomAccessRange, typename Qubit, typename Function>
          [[noreturn]] inline RandomAccessRange& one_page_qubit_gate(
            ParallelPolicy const,
            RandomAccessRange&, ::ket::mpi::permutated<Qubit> const, Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"one_page_qubit_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, typename Allocator, typename Qubit, typename Function>
          [[noreturn]] inline ::ket::mpi::state<Complex, 0, Allocator>& one_page_qubit_gate(
            ParallelPolicy const,
            ::ket::mpi::state<Complex, 0, Allocator>&, ::ket::mpi::permutated<Qubit> const, Function&&)
          { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"one_page_qubit_gate"}; }

          template <
            std::size_t num_operated_nonpage_qubits,
            typename ParallelPolicy,
            typename Complex, int num_page_qubits_, typename Allocator, typename Qubit, typename Function>
          inline ::ket::mpi::state<Complex, num_page_qubits_, Allocator>& one_page_qubit_gate(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::state<Complex, num_page_qubits_, Allocator>& local_state,
            ::ket::mpi::permutated<Qubit> const permutated_qubit,
            Function&& function)
          {
            static_assert(
              num_page_qubits_ >= 1,
              "num_page_qubits_ should be greater than or equal to 1");
            assert(::ket::mpi::page::is_on_page(permutated_qubit, local_state));

            using bit_integer_type = typename ::ket::meta::bit_integer_of<Qubit>::type;
            using state_integer_type = typename ::ket::meta::state_integer_of<Qubit>::type;
            auto const num_nonpage_local_qubits
              = static_cast<bit_integer_type>(local_state.num_local_qubits() - num_page_qubits_);
            auto const permutated_qubit_mask
              = ::ket::utility::integer_exp2<state_integer_type>(permutated_qubit - num_nonpage_local_qubits);
            auto const lower_bits_mask = permutated_qubit_mask - state_integer_type{1u};
            auto const upper_bits_mask = compl lower_bits_mask;

            static constexpr auto num_pages
              = ::ket::mpi::state<Complex, num_page_qubits_, Allocator>::num_pages;
            auto const num_data_blocks = local_state.num_data_blocks();
            for (auto data_block_index = std::size_t{0u}; data_block_index < num_data_blocks; ++data_block_index)
              for (auto page_index_wo_qubit = std::size_t{0u}; page_index_wo_qubit < num_pages / 2u; ++page_index_wo_qubit)
              {
                // x0x
                auto const zero_page_index
                  = ((page_index_wo_qubit bitand upper_bits_mask) << 1u)
                    bitor (page_index_wo_qubit bitand lower_bits_mask);
                // x1x
                auto const one_page_index = zero_page_index bitor permutated_qubit_mask;

                auto zero_page_range = local_state.page_range(std::make_pair(data_block_index, zero_page_index));
                auto one_page_range = local_state.page_range(std::make_pair(data_block_index, one_page_index));
                assert(boost::size(zero_page_range) == boost::size(one_page_range));
                assert(::ket::utility::integer_exp2<std::size_t>(::ket::utility::integer_log2<std::size_t>(boost::size(zero_page_range))) == boost::size(zero_page_range));

                auto const zero_first = std::begin(zero_page_range);
                auto const one_first = std::begin(one_page_range);

                using ::ket::utility::loop_n;
                loop_n(
                  parallel_policy,
                  boost::size(zero_page_range) >> num_operated_nonpage_qubits,
                  [zero_first, one_first, &function](state_integer_type const index_wo_nonpage_qubits, int const thread_index)
                  { function(zero_first, one_first, index_wo_nonpage_qubits, thread_index); });
              }

            return local_state;
          }
        } // namespace detail
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_DETAIL_ONE_PAGE_QUBIT_GATE_HPP
