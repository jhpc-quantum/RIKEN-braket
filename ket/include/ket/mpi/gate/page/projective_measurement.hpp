#ifndef KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP
# define KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP

# include <boost/config.hpp>

# include <cmath>
# include <utility>

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/state.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // zero_one_probabilities
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<RandomAccessRange>::type>::type,
          typename ::ket::utility::meta::real_of<
            typename boost::range_value<RandomAccessRange>::type>::type>
        zero_one_probabilities(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange const& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        {
          typedef
            typename ::ket::utility::meta::real_of<
              typename boost::range_value<RandomAccessRange>::type>::type
            result_type;
          return std::make_pair(result_type(0), result_type(0));
        }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<Complex>::type,
          typename ::ket::utility::meta::real_of<Complex>::type>
        zero_one_probabilities(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator> const& local_state,
          ::ket::qubit<StateInteger, BitInteger> const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        {
          typedef
            typename ::ket::utility::meta::real_of<Complex>::type
            result_type;
          return std::make_pair(result_type(0), result_type(0));
        }

        namespace projective_measurement_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct zero_one_probabilities
          {
            long double& zero_probability_;
            long double& one_probability_;

            zero_one_probabilities(
              long double& zero_probability, long double& one_probability)
              : zero_probability_{zero_probability},
                one_probability_{one_probability}
            { }

            template <typename Iterator>
            void operator()(Iterator const zero_iter, Iterator const one_iter) const
            {
              using std::norm;
              zero_probability_ += static_cast<long double>(norm(*zero_iter));
              one_probability_ += static_cast<long double>(norm(*one_iter));
            }
          }; // struct zero_one_probabilities
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace projective_measurement_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename PermutationAllocator>
        inline
        std::pair<
          typename ::ket::utility::meta::real_of<Complex>::type,
          typename ::ket::utility::meta::real_of<Complex>::type>
        zero_one_probabilities(
          ::ket::mpi::utility::policy::general_mpi const mpi_policy,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          auto zero_probability = 0.0l;
          auto one_probability = 0.0l;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [&zero_probability, &one_probability](
              auto const zero_iter, auto const one_iter)
            {
              using std::norm;
              zero_probability += static_cast<long double>(norm(*zero_iter));
              one_probability += static_cast<long double>(norm(*one_iter));
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::projective_measurement_detail::zero_one_probabilities{
              zero_probability, one_probability});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          using real_type = typename ::ket::utility::meta::real_of<Complex>::type;
          return std::make_pair(
            static_cast<real_type>(zero_probability), static_cast<real_type>(one_probability));
        }

        // change_state_after_measuring_zero
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Real, typename Allocator>
        inline void change_state_after_measuring_zero(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_zero(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { }

        namespace projective_measurement_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct change_state_after_measuring_zero
          {
            Real multiplier_;

            explicit change_state_after_measuring_zero(Real const& multiplier)
              : multiplier_{multiplier}
            { }

            template <typename Iterator>
            void operator()(Iterator const zero_iter, Iterator const one_iter) const
            {
              *zero_iter *= multiplier_;
              *one_iter = Complex{Real{0}};
            }
          }; // struct change_state_after_measuring_zero<Complex, Real>

          template <typename Complex, typename Real>
          inline ::ket::mpi::gate::page::projective_measurement_detail::change_state_after_measuring_zero<Complex, Real>
          make_change_state_after_measuring_zero(Real const& multiplier)
          { return ::ket::mpi::gate::page::projective_measurement_detail::change_state_after_measuring_zero<Complex, Real>{multiplier}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace projective_measurement_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_zero(
          ::ket::mpi::utility::policy::general_mpi const mpi_policy,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          Real const zero_probability,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(zero_probability, -half<Real>());

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [multiplier](auto const zero_iter, auto const one_iter)
            {
              *zero_iter *= multiplier;
              *one_iter = Complex{Real{0}};
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::projective_measurement_detail::make_change_state_after_measuring_zero<Complex>(multiplier));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // change_state_after_measuring_one
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Real, typename Allocator>
        inline void change_state_after_measuring_one(
          MpiPolicy const, ParallelPolicy const,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const&)
        { }

        template <
          typename ParallelPolicy,
          typename Complex, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_one(
          ::ket::mpi::utility::policy::general_mpi const, ParallelPolicy const,
          ::ket::mpi::state<Complex, 0, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const, Real const,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&)
        { }

        namespace projective_measurement_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct change_state_after_measuring_one
          {
            Real multiplier_;

            explicit change_state_after_measuring_one(Real const& multiplier)
              : multiplier_{multiplier}
            { }

            template <typename Iterator>
            void operator()(Iterator const zero_iter, Iterator const one_iter) const
            {
              *zero_iter = Complex{Real{0}};
              *one_iter *= multiplier_;
            }
          }; // struct change_state_after_measuring_one<Complex, Real>

          template <typename Complex, typename Real>
          inline ::ket::mpi::gate::page::projective_measurement_detail::change_state_after_measuring_one<Complex, Real>
          make_change_state_after_measuring_one(Real const& multiplier)
          { return ::ket::mpi::gate::page::projective_measurement_detail::change_state_after_measuring_one<Complex, Real>{multiplier}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace projective_measurement_detail

        template <
          typename ParallelPolicy,
          typename Complex, int num_page_qubits_, typename StateAllocator,
          typename StateInteger, typename BitInteger, typename Real, typename PermutationAllocator>
        inline void change_state_after_measuring_one(
          ::ket::mpi::utility::policy::general_mpi const mpi_policy,
          ParallelPolicy const parallel_policy,
          ::ket::mpi::state<Complex, num_page_qubits_, StateAllocator>& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          Real const one_probability,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, PermutationAllocator> const&
            permutation)
        {
          using std::pow;
          using boost::math::constants::half;
          auto const multiplier = pow(one_probability, -half<Real>());

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [multiplier](auto const zero_iter, auto const one_iter)
            {
              *zero_iter = Complex{Real{0}};
              *one_iter *= multiplier;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::gate::page::detail::one_page_qubit_gate(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::projective_measurement_detail::make_change_state_after_measuring_one<Complex>(multiplier));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_PROJECTIVE_MEASUREMENT_HPP
