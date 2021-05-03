#ifndef KET_MPI_GATE_PAGE_SET_HPP
# define KET_MPI_GATE_PAGE_SET_HPP

# include <cassert>

# include <boost/math/constants/constants.hpp>
# include <boost/range/size.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/meta/real_of.hpp>
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
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& set(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { return local_state; }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, 0, StateAllocator>& set(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { return local_state; }

        namespace set_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct set1
          {
            Real& one_probability_;

            explicit set1(Real& one_probability)
              : one_probability_{one_probability}
            { }

            template <typename Iterator>
            void operator()(Iterator const zero_iter, Iterator const one_iter) const
            {
              *zero_iter = Complex{0};

              using std::norm;
              one_probability_ += norm(*one_iter);
            }
          }; // struct set1<Complex, Real>

          template <typename Complex, typename Real>
          inline ::ket::mpi::gate::page::set_detail::set1<Complex, Real>
          make_set1(Real& one_probability)
          { return ::ket::mpi::gate::page::set_detail::set1<Complex, Real>{one_probability}; }

          template <typename Real>
          struct set2
          {
            Real multiplier_;

            explicit set2(Real const multiplier)
              : multiplier_{multiplier}
            { }

            template <typename Iterator>
            void operator()(Iterator const, Iterator const one_iter) const
            { *one_iter *= multiplier_; }
          }; // struct set2<Complex, Real>

          template <typename Real>
          inline ::ket::mpi::gate::page::set_detail::set2<Real>
          make_set2(Real const multiplier)
          { return ::ket::mpi::gate::page::set_detail::set2<Real>{multiplier}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace set_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& set(
          ::ket::mpi::utility::policy::general_mpi const mpi_policy,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto one_probability = real_type{0};

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [&one_probability](auto const zero_iter, auto const one_iter)
            {
              *zero_iter = Complex{0};

              using std::norm;
              one_probability += norm(*one_iter);
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::set_detail::make_set1<Complex>(one_probability));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(one_probability, -half<real_type>());

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [multiplier](auto const, auto const one_iter)
            { *one_iter *= multiplier; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::set_detail::make_set2(multiplier));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_SET_HPP
