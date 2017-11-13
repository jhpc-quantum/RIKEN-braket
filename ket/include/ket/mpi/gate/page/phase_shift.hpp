#ifndef KET_MPI_GATE_PAGE_PHASE_SHIFT_HPP
# define KET_MPI_GATE_PAGE_PHASE_SHIFT_HPP

# include <boost/config.hpp>

# include <cassert>
# include <iterator>

# include <boost/math/constants/constants.hpp>
# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/exp_i.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // phase_shift_coeff
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& phase_shift_coeff(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& phase_shift_coeff(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          Complex const&,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }


        namespace phase_shift_detail
        {
# ifdef BOOST_NO_CXX11_LAMBDAS
          template <typename RandomAccessIterator, typename Complex>
          struct phase_shift_coeff_loop_inside
          {
            RandomAccessIterator first_;
            Complex phase_coefficient_;

            phase_shift_coeff_loop_inside(
              RandomAccessIterator const first, Complex const& phase_coefficient)
              : first_(first), phase_coefficient_(phase_coefficient)
            { }

            template <typename StateInteger>
            void operator()(StateInteger const index, int const) const
            { *(first_+index) *= phase_coefficient_; }
          };

          template <typename RandomAccessIterator, typename Complex>
          inline phase_shift_coeff_loop_inside<RandomAccessIterator, Complex>
          make_phase_shift_coeff_loop_inside(
            RandomAccessIterator const first, Complex const& phase_coefficient)
          {
            typedef
              ::ket::mpi::gate::page::phase_shift_detail
                ::phase_shift_coeff_loop_inside<RandomAccessIterator, Complex>
              result_type;

            return result_type(first, phase_coefficient);
          }
# endif // BOOST_NO_CXX11_LAMBDAS
        } // namespace phase_shift_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& phase_shift_coeff(
          ::ket::mpi::utility::policy::general_mpi const,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          Complex const& phase_coefficient,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          assert(local_state.is_page_qubit(permutation[qubit]));

          typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

          BitInteger const num_nonpage_qubits
            = static_cast<BitInteger>(local_state.num_local_qubits()-num_page_qubits_);
          StateInteger const qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(
                permutation[qubit] - static_cast<qubit_type>(num_nonpage_qubits));
          StateInteger const lower_bits_mask = qubit_mask-static_cast<StateInteger>(1u);
          StateInteger const upper_bits_mask = compl lower_bits_mask;

          typedef ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator> local_state_type;
          for (std::size_t base_page_id = 0u;
               base_page_id < local_state_type::num_pages/2u; ++base_page_id)
          {
            // x0x
            StateInteger const zero_page_id
              = ((base_page_id bitand upper_bits_mask) << 1u)
                bitor (base_page_id bitand lower_bits_mask);
            // x1x
            StateInteger const one_page_id = zero_page_id bitor qubit_mask;

            typedef typename local_state_type::page_range_type page_range_type;
            page_range_type one_page_range
              = local_state.page_range(one_page_id);

            using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(one_page_range),
              [&one_page_range, phase_coefficient](StateInteger const index, int const)
              { *(boost::begin(one_page_range)+index) *= phase_coefficient; });
# else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy,
              boost::size(one_page_range),
              ::ket::mpi::gate::page::phase_shift_detail::make_phase_shift_coeff_loop_inside(
                boost::begin(one_page_range), phase_coefficient));
# endif // BOOST_NO_CXX11_LAMBDAS
          }

          return local_state;
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif

