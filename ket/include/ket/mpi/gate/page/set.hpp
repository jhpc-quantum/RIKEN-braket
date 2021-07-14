#ifndef KET_MPI_GATE_PAGE_SET_HPP
# define KET_MPI_GATE_PAGE_SET_HPP

# include <boost/config.hpp>

# include <cassert>

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
        [[noreturn]] inline RandomAccessRange& set(
          ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"set"}; }

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        [[noreturn]] inline ::ket::mpi::state<Complex, false, Allocator>& set(
          ParallelPolicy const,
          ::ket::mpi::state<Complex, false, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const)
        { throw ::ket::mpi::gate::page::unsupported_page_gate_operation{"set"}; }

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

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              *(zero_first + index) = Complex{0};

              using std::norm;
              one_probability_ += norm(*(one_first + index));
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

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const, Iterator const one_first, StateInteger const index, int const) const
            { *(one_first + index) *= multiplier_; }
          }; // struct set2<Complex, Real>

          template <typename Real>
          inline ::ket::mpi::gate::page::set_detail::set2<Real>
          make_set2(Real const multiplier)
          { return ::ket::mpi::gate::page::set_detail::set2<Real>{multiplier}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace set_detail

        template <
          typename ParallelPolicy,
          typename Complex, typename Allocator, typename StateInteger, typename BitInteger>
        inline ::ket::mpi::state<Complex, true, Allocator>& set(
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, true, Allocator>& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          auto one_probability = real_type{0};

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [&one_probability](auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              *(zero_first + index) = Complex{0};

              using std::norm;
              one_probability += norm(*(one_first + index));
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::set_detail::make_set1<Complex>(one_probability));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(one_probability, -half<real_type>());

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [multiplier](auto const, auto const one_first, StateInteger const index, int const)
            { *(one_first + index) *= multiplier; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::set_detail::make_set2(multiplier));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_SET_HPP
