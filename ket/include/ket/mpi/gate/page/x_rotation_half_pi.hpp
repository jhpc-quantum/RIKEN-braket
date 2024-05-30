#ifndef KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP
# define KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP

# include <boost/config.hpp>

# include <cassert>

# include <boost/math/constants/constants.hpp>
# include <boost/range/value_type.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // +X_i
        // +X_1 (a_0 |0> + a_1 |1>) = (a_0 + i a_1)/sqrt(2) |0> + (i a_0 + a_1)/sqrt(2) |1>
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct x_rotation_half_pi
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
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
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& x_rotation_half_pi(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [](auto const zero_first, auto const one_first, StateInteger const index, int const)
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
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::x_rotation_half_pi<complex_type, real_type>{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cx_rotation_half_pi_tcp: both of target and control qubits of CH are on page
        // C+X_{tc} or C1+X_{tc}
        // C+X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + i a_{11})/sqrt(2) |10> + (i a_{10} + a_{11})/sqrt(2) |11>
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct cx_rotation_half_pi_tcp
          {
            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            }
          }; // struct cx_rotation_half_pi_tcp
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace x_rotation_half_pi_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cx_rotation_half_pi_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [](auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::cx_rotation_half_pi_tcp{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cx_rotation_half_pi_tp: only target qubit is on page
        // C+X_{tc} or C1+X_{tc}
        // C+X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + i a_{11})/sqrt(2) |10> + (i a_{10} + a_{11})/sqrt(2) |11>
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct cx_rotation_half_pi_tp
          {
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            cx_rotation_half_pi_tp(
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : control_qubit_mask_{control_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const zero_first, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor control_qubit_mask_;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            }
          }; // struct cx_rotation_half_pi_tp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::x_rotation_half_pi_detail::cx_rotation_half_pi_tp<StateInteger>
          make_cx_rotation_half_pi_tp(
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace x_rotation_half_pi_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cx_rotation_half_pi_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::make_cx_rotation_half_pi_tp(
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cx_rotation_half_pi_cp: only control qubit is on page
        // C+X_{tc} or C1+X_{tc}
        // C+X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} + i a_{11})/sqrt(2) |10> + (i a_{10} + a_{11})/sqrt(2) |11>
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct cx_rotation_half_pi_cp
          {
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            cx_rotation_half_pi_cp(
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : target_qubit_mask_{target_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor target_qubit_mask_;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            }
          }; // struct cx_rotation_half_pi_cp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::x_rotation_half_pi_detail::cx_rotation_half_pi_cp<StateInteger>
          make_cx_rotation_half_pi_cp(
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace x_rotation_half_pi_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& cx_rotation_half_pi_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter += ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::make_cx_rotation_half_pi_cp(
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename Real>
          struct adj_x_rotation_half_pi
          {
            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
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
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_x_rotation_half_pi(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [](auto const zero_first, auto const one_first, StateInteger const index, int const)
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
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::adj_x_rotation_half_pi<complex_type, real_type>{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cx_rotation_half_pi_tcp: both of target and control qubits of CH are on page
        // C-X_{tc} or C1-X_{tc}
        // C-X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - i a_{11})/sqrt(2) |10> + (-i a_{10} + a_{11})/sqrt(2) |11>
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          struct adj_cx_rotation_half_pi_tcp
          {
            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            }
          }; // struct adj_cx_rotation_half_pi_tcp
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace x_rotation_half_pi_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cx_rotation_half_pi_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [](auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              auto const control_on_iter = first_10 + index;
              auto const target_control_on_iter = first_11 + index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::adj_cx_rotation_half_pi_tcp{});
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cx_rotation_half_pi_tp: only target qubit is on page
        // C-X_{tc} or C1-X_{tc}
        // C-X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - i a_{11})/sqrt(2) |10> + (-i a_{10} + a_{11})/sqrt(2) |11>
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct adj_cx_rotation_half_pi_tp
          {
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            adj_cx_rotation_half_pi_tp(
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : control_qubit_mask_{control_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const zero_first, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor control_qubit_mask_;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            }
          }; // struct adj_cx_rotation_half_pi_tp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::x_rotation_half_pi_detail::adj_cx_rotation_half_pi_tp<StateInteger>
          make_adj_cx_rotation_half_pi_tp(
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace x_rotation_half_pi_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cx_rotation_half_pi_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              auto const control_on_iter = zero_first + one_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::make_adj_cx_rotation_half_pi_tp(
              control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // adj_cx_rotation_half_pi_cp: only control qubit is on page
        // C-X_{tc} or C1-X_{tc}
        // C-X_{1,2} (a_{00} |00> + a_{01} |01> + a_{10} |10> + a_{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + (a_{10} - i a_{11})/sqrt(2) |10> + (-i a_{10} + a_{11})/sqrt(2) |11>
        namespace x_rotation_half_pi_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename StateInteger>
          struct adj_cx_rotation_half_pi_cp
          {
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            adj_cx_rotation_half_pi_cp(
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : target_qubit_mask_{target_qubit_mask},
                nonpage_lower_bits_mask_{nonpage_lower_bits_mask},
                nonpage_upper_bits_mask_{nonpage_upper_bits_mask}
            { }

            template <typename Iterator>
            void operator()(
              Iterator const, Iterator const one_first,
              StateInteger const index_wo_nonpage_qubit, int const) const
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask_) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask_);
              auto const one_index = zero_index bitor target_qubit_mask_;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            }
          }; // struct adj_cx_rotation_half_pi_cp<StateInteger>

          template <typename StateInteger>
          inline ::ket::mpi::gate::page::x_rotation_half_pi_detail::adj_cx_rotation_half_pi_cp<StateInteger>
          make_adj_cx_rotation_half_pi_cp(
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace x_rotation_half_pi_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& adj_cx_rotation_half_pi_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              auto const control_on_iter = one_first + zero_index;
              auto const target_control_on_iter = one_first + one_index;
              auto const control_on_iter_value = *control_on_iter;

              using complex_type = typename std::remove_const<decltype(control_on_iter_value)>::type;
              using real_type = typename ::ket::utility::meta::real_of<complex_type>::type;
              using boost::math::constants::one_div_root_two;
              *control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * (*target_control_on_iter);
              *control_on_iter *= one_div_root_two<real_type>();
              *target_control_on_iter -= ::ket::utility::imaginary_unit<complex_type>() * control_on_iter_value;
              *target_control_on_iter *= one_div_root_two<real_type>();
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::x_rotation_half_pi_detail::make_adj_cx_rotation_half_pi_cp(
              target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_X_ROTATION_HALF_PI_HPP
