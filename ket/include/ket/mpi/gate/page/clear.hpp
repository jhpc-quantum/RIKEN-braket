#ifndef KET_MPI_GATE_PAGE_CLEAR_HPP
# define KET_MPI_GATE_PAGE_CLEAR_HPP

# include <boost/config.hpp>

# include <cmath>

# include <boost/math/constants/constants.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/gate/page/unsupported_page_gate_operation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


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
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        [[noreturn]] inline RandomAccessRange& clear(
          ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0, false>{"clear"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        [[noreturn]] inline ::ket::mpi::state<Complex, 0, Allocator>& clear(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation<0>{"clear"}; }

        namespace clear_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct clear1
          {
            Real& zero_probability_;

            explicit clear1(Real& zero_probability)
              : zero_probability_{zero_probability}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              *(one_first + index) = Complex{0};

              using std::norm;
              zero_probability_ += norm(*(zero_first + index));
            }
          }; // struct clear1<Complex, Real>

          template <typename Complex, typename Real>
          inline ::ket::mpi::gate::page::clear_detail::clear1<Complex, Real>
          make_clear1(Real& zero_probability)
          { return ::ket::mpi::gate::page::clear_detail::clear1<Complex, Real>{zero_probability}; }

          template <typename Real>
          struct clear2
          {
            Real multiplier_;

            explicit clear2(Real const multiplier)
              : multiplier_{multiplier}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const, StateInteger const index, int const) const
            { *(zero_first + index) *= multiplier_; }
          }; // struct clear2<Complex, Real>

          template <typename Real>
          inline ::ket::mpi::gate::page::clear_detail::clear2<Real>
          make_clear2(Real const multiplier)
          { return ::ket::mpi::gate::page::clear_detail::clear2<Real>{multiplier}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace clear_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename Allocator,
          typename StateInteger, typename BitInteger>
        inline ::ket::mpi::state<Complex, num_page_qubits_, Allocator>& clear(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto zero_probability = real_type{0};

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [&zero_probability](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              *(one_first + index) = Complex{0};

              using std::norm;
              zero_probability += norm(*(zero_first + index));
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::clear_detail::make_clear1<Complex>(zero_probability));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(zero_probability, -half<real_type>());

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [multiplier](auto const zero_first, auto const, StateInteger const index, int const)
            { *(zero_first + index) *= multiplier; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::clear_detail::make_clear2(multiplier));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_CLEAR_HPP
