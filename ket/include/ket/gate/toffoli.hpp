#ifndef KET_GATE_TOFFOLI_HPP
# define KET_GATE_TOFFOLI_HPP

# include <cassert>
# include <array>
# include <iterator>
# include <algorithm>
# include <utility>
# include <type_traits>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/integer_log2.hpp>
# endif


namespace ket
{
  namespace gate
  {
    // TOFFOLI_{t,c1,c2}
    // TOFFOLI_{1,2,3} (a_{000} |000> + a_{001} |001> + a_{010} |010> + a_{011} |011> + a_{100} |100> + a_{101} |101> + a_{110} |110> + a_{111} |111>)
    //   = a_{000} |000> + a_{001} |001> + a_{010} |010> + a_{011} |011> + a_{100} |100> + a_{101} |101> + a_{111} |110> + a_{110} |111>
    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto toffoli(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    {
      static_assert(std::is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(std::is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(
        ::ket::utility::integer_exp2<StateInteger>(target_qubit) < static_cast<StateInteger>(last - first)
        and ::ket::utility::integer_exp2<StateInteger>(control_qubit1) < static_cast<StateInteger>(last - first)
        and ::ket::utility::integer_exp2<StateInteger>(control_qubit2) < static_cast<StateInteger>(last - first)
        and target_qubit != control_qubit1 and target_qubit != control_qubit2 and control_qubit1 != control_qubit2);
      assert(
        ::ket::utility::integer_exp2<StateInteger>(::ket::utility::integer_log2<BitInteger>(last - first))
        == static_cast<StateInteger>(last - first));

      using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
      std::array<qubit_type, 3u> sorted_qubits{target_qubit, control_qubit1.qubit(), control_qubit2.qubit()};
      using std::begin;
      using std::end;
      std::sort(begin(sorted_qubits), end(sorted_qubits));

      auto const target_qubit_mask = ::ket::utility::integer_exp2<StateInteger>(target_qubit);
      auto const control_qubits_mask
        = ::ket::utility::integer_exp2<StateInteger>(control_qubit1)
          bitor ::ket::utility::integer_exp2<StateInteger>(control_qubit2);

      std::array<StateInteger, 4u> bits_mask{};
      bits_mask[0u] = ::ket::utility::integer_exp2<StateInteger>(sorted_qubits[0u]) - StateInteger{1u};
      bits_mask[1u]
        = (::ket::utility::integer_exp2<StateInteger>(sorted_qubits[1u] - BitInteger{1u}) - StateInteger{1u})
          xor bits_mask[0u];
      bits_mask[2u]
        = (::ket::utility::integer_exp2<StateInteger>(sorted_qubits[2u] - BitInteger{2u}) - StateInteger{1u})
          xor (bits_mask[0u] bitor bits_mask[1u]);
      bits_mask[3u] = compl (bits_mask[0u] bitor bits_mask[1u] bitor bits_mask[2u]);

      ::ket::utility::loop_n(
        parallel_policy, static_cast<StateInteger>(last - first) >> 3u,
        [first, target_qubit_mask, control_qubits_mask, &bits_mask](StateInteger const value_wo_qubits, int const)
        {
          // xxx0_cxxx0_txxxx0_cxxx
          auto const base_index
            = ((value_wo_qubits bitand bits_mask[3u]) << 3u)
              bitor ((value_wo_qubits bitand bits_mask[2u]) << 2u)
              bitor ((value_wo_qubits bitand bits_mask[1u]) << 1u)
              bitor (value_wo_qubits bitand bits_mask[0u]);
          // xxx1_cxxx0_txxxx1_cxxx
          auto const control_on_index = base_index bitor control_qubits_mask;
          // xxx1_cxxx1_txxxx1_cxxx
          auto const target_control_on_index = control_on_index bitor target_qubit_mask;

          std::iter_swap(first + control_on_index, first + target_control_on_index);
        });
    }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto toffoli(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    { ::ket::gate::toffoli(::ket::utility::policy::make_sequential(), first, last, target_qubit, control_qubit1, control_qubit2); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline auto toffoli(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> RandomAccessRange&
      {
        using std::begin;
        using std::end;
        ::ket::gate::toffoli(parallel_policy, begin(state), end(state), target_qubit, control_qubit1, control_qubit2);
        return state;
      }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline auto toffoli(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::toffoli(::ket::utility::policy::make_sequential(), state, target_qubit, control_qubit1, control_qubit2); }
    } // namespace ranges

    template <typename ParallelPolicy, typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_toffoli(
      ParallelPolicy const parallel_policy,
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    { ::ket::gate::toffoli(parallel_policy, first, last, target_qubit, control_qubit1, control_qubit2); }

    template <typename RandomAccessIterator, typename StateInteger, typename BitInteger>
    inline auto adj_toffoli(
      RandomAccessIterator const first, RandomAccessIterator const last,
      ::ket::qubit<StateInteger, BitInteger> const target_qubit,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
      ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
    -> void
    { ::ket::gate::toffoli(first, last, target_qubit, control_qubit1, control_qubit2); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline auto adj_toffoli(
        ParallelPolicy const parallel_policy, RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::toffoli(parallel_policy, state, target_qubit, control_qubit1, control_qubit2); }

      template <typename RandomAccessRange, typename StateInteger, typename BitInteger>
      inline auto adj_toffoli(
        RandomAccessRange& state,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit1,
        ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit2)
      -> RandomAccessRange&
      { return ::ket::gate::ranges::toffoli(state, target_qubit, control_qubit1, control_qubit2); }
    } // namespace ranges
  } // namespace gate
} // namespace ket


#endif // KET_GATE_TOFFOLI_HPP
