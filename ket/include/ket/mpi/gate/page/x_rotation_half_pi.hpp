#ifndef KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP
# define KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP

# include <boost/config.hpp>

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>

# include <ket/qubit.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct x_rotation_half_pi
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using boost::math::constants::one_div_root_two;
              *zero_iter += ::ket::utility::imaginary_unit<Complex>() * *one_iter;
              *zero_iter *= one_div_root_two<Real>();
              *one_iter += ::ket::utility::imaginary_unit<Complex>() * zero_iter_value;
              *one_iter *= one_div_root_two<Real>();
            }
          }; // struct x_rotation_half_pi<Complex, Real>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace x_rotation_half_pi_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& x_rotation_half_pi(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [](auto const zero_first, auto const one_first, StateInteger const index)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using boost::math::constants::one_div_root_two;
              *zero_iter += ::ket::utility::imaginary_unit<complex_type>() * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter += ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
              *one_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::x_rotation_half_pi<complex_type, real_type>{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct adj_x_rotation_half_pi
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index) const
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using boost::math::constants::one_div_root_two;
              *zero_iter -= ::ket::utility::imaginary_unit<Complex>() * *one_iter;
              *zero_iter *= one_div_root_two<Real>();
              *one_iter -= ::ket::utility::imaginary_unit<Complex>() * zero_iter_value;
              *one_iter *= one_div_root_two<Real>();
            }
          }; // struct adj_x_rotation_half_pi<Complex, Real>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace adj_x_rotation_half_pi_detail

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<
            StateInteger, BitInteger, Allocator> const& permutation)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            [](auto const zero_first, auto const one_first, StateInteger const index)
            {
              auto const zero_iter = zero_first + index;
              auto const one_iter = one_first + index;
              auto const zero_iter_value = *zero_iter;

              using boost::math::constants::one_div_root_two;
              *zero_iter -= ::ket::utility::imaginary_unit<complex_type>() * *one_iter;
              *zero_iter *= one_div_root_two<real_type>();
              *one_iter -= ::ket::utility::imaginary_unit<complex_type>() * zero_iter_value;
              *one_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            mpi_policy, parallel_policy, local_state, qubit, permutation,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::adj_x_rotation_half_pi<complex_type, real_type>{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP
