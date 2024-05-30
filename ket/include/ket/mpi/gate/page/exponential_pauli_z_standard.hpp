#ifndef KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_Z_STANDARD_HPP
# define KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_Z_STANDARD_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cmath>
# include <type_traits>
# include <memory>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/imaginary_unit.hpp>
# include <ket/mpi/permutated.hpp>
# include <ket/mpi/page/is_on_page.hpp>
# include <ket/mpi/gate/page/detail/two_page_qubits_gate.hpp>
# include <ket/mpi/gate/page/detail/one_page_qubit_gate.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      namespace page
      {
        // 1_p: the qubit of eY is on page
        // eZ_i(theta) = exp(i theta Z_i) = I cos(theta) + i Z_i sin(theta)
        // eZ_1(theta) (a_0 |0> + a_1 |1>) = e^{i theta} a_0 |0> + e^{-i theta} a_1 |1>
        namespace exponential_pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct exponential_pauli_z_coeff1
          {
            Complex phase_coefficient_;
            Complex conj_phase_coefficient_;

            exponential_pauli_z_coeff1(
              Complex const& phase_coefficient, Complex const& conj_phase_coefficient) noexcept
              : phase_coefficient_{phase_coefficient}, conj_phase_coefficient_{conj_phase_coefficient}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(Iterator const zero_first, Iterator const one_first, StateInteger const index, int const) const
            {
              *(zero_first + index) *= phase_coefficient_;
              *(one_first + index) *= conj_phase_coefficient_;
            }
          }; // struct exponential_pauli_z_coeff1<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::exponential_pauli_z_detail::exponential_pauli_z_coeff1<Complex>
          make_exponential_pauli_z_coeff1(Complex const& phase_coefficient, Complex const& conj_phase_coefficient)
          { return {phase_coefficient, conj_phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_z_coeff1(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit)
        {
          assert(::ket::mpi::page::is_on_page(permutated_qubit, local_state));

          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<0u>(
            parallel_policy, local_state, permutated_qubit,
            [&phase_coefficient, &conj_phase_coefficient](
              auto const zero_first, auto const one_first, StateInteger const index, int const)
            {
              *(zero_first + index) *= phase_coefficient;
              *(one_first + index) *= conj_phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_qubit,
            ::ket::mpi::gate::page::exponential_pauli_z_detail::make_exponential_pauli_z_coeff1(
              phase_coefficient, conj_phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_2p: both of qubits of eYY are on page
        // eZZ_{ij}(theta) = exp(i theta Z_i Z_j) = I cos(theta) + i Z_i Z_j sin(theta)
        // eZZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = e^{i theta} |00> + e^{-i theta} |01> + e^{-i theta} |10> + e^{i theta} |11>
        namespace exponential_pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct exponential_pauli_z_coeff2_2p
          {
            Complex phase_coefficient_;
            Complex conj_phase_coefficient_;

            exponential_pauli_z_coeff2_2p(Complex const& phase_coefficient, Complex const& conj_phase_coefficient) noexcept
              : phase_coefficient_{phase_coefficient}, conj_phase_coefficient_{conj_phase_coefficient}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const first_00, Iterator const first_01,
              Iterator const first_10, Iterator const first_11,
              StateInteger const index, int const) const
            {
              *(first_00 + index) *= phase_coefficient_;
              *(first_01 + index) *= conj_phase_coefficient_;
              *(first_10 + index) *= conj_phase_coefficient_;
              *(first_11 + index) *= phase_coefficient_;
            }
          }; // struct exponential_pauli_z_coeff2_2p<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::exponential_pauli_z_detail::exponential_pauli_z_coeff2_2p<Complex>
          make_exponential_pauli_z_coeff2_2p(Complex const& phase_coefficient, Complex const& conj_phase_coefficient)
          { return {phase_coefficient, conj_phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_z_coeff2_2p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit1,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit2)
        {
          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            [&phase_coefficient, &conj_phase_coefficient](
              auto const first_00, auto const first_01, auto const first_10, auto const first_11,
              StateInteger const index, int const)
            {
              *(first_00 + index) *= phase_coefficient;
              *(first_01 + index) *= conj_phase_coefficient;
              *(first_10 + index) *= conj_phase_coefficient;
              *(first_11 + index) *= phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, page_permutated_qubit1, page_permutated_qubit2,
            ::ket::mpi::gate::page::exponential_pauli_z_detail::make_exponential_pauli_z_coeff2_2p(
              phase_coefficient, conj_phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // 2_p: only one qubit of eYY is on page
        // eZZ_{ij}(theta) = exp(i theta Z_i Z_j) = I cos(theta) + i Z_i Z_j sin(theta)
        // eZZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = e^{i theta} |00> + e^{-i theta} |01> + e^{-i theta} |10> + e^{i theta} |11>
        namespace exponential_pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct exponential_pauli_z_coeff2_p
          {
            Complex phase_coefficient_;
            Complex conj_phase_coefficient_;
            StateInteger nonpage_permutated_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            exponential_pauli_z_coeff2_p(
              Complex const& phase_coefficient, Complex const& conj_phase_coefficient,
              StateInteger const nonpage_permutated_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : phase_coefficient_{phase_coefficient}, conj_phase_coefficient_{conj_phase_coefficient},
                nonpage_permutated_qubit_mask_{nonpage_permutated_qubit_mask},
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
              auto const one_index = zero_index bitor nonpage_permutated_qubit_mask_;

              auto const iter_00 = zero_first + zero_index;
              auto const iter_01_or_10 = zero_first + one_index;
              auto const iter_10_or_01 = one_first + zero_index;
              auto const iter_11 = one_first + one_index;

              *iter_00 *= phase_coefficient_;
              *iter_01_or_10 *= conj_phase_coefficient_;
              *iter_10_or_01 *= conj_phase_coefficient_;
              *iter_11 *= phase_coefficient_;
            }
          }; // struct exponential_pauli_z_coeff2_p<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::exponential_pauli_z_detail::exponential_pauli_z_coeff2_p<Complex, StateInteger>
          make_exponential_pauli_z_coeff2_p(
            Complex const& phase_coefficient, Complex const& conj_phase_coefficient,
            StateInteger const nonpage_permutated_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient, conj_phase_coefficient, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_z_coeff2_p(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const page_permutated_qubit,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const nonpage_permutated_qubit)
        {
          assert(::ket::mpi::page::is_on_page(page_permutated_qubit, local_state));
          assert(not ::ket::mpi::page::is_on_page(nonpage_permutated_qubit, local_state));

          auto const nonpage_permutated_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(nonpage_permutated_qubit);
          auto const nonpage_lower_bits_mask = nonpage_permutated_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            [&phase_coefficient, &conj_phase_coefficient, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor nonpage_permutated_qubit_mask;

              auto const iter_00 = zero_first + zero_index;
              auto const iter_01_or_10 = zero_first + one_index;
              auto const iter_10_or_01 = one_first + zero_index;
              auto const iter_11 = one_first + one_index;

              *iter_00 *= phase_coefficient;
              *iter_01_or_10 *= conj_phase_coefficient;
              *iter_10_or_01 *= conj_phase_coefficient;
              *iter_11 *= phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, page_permutated_qubit,
            ::ket::mpi::gate::page::exponential_pauli_z_detail::make_exponential_pauli_z_coeff2_p(
              phase_coefficient, conj_phase_coefficient, nonpage_permutated_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cz_coeff_tcp: both of target and control qubits of CeZ are on page
        // CeZ_{tc}(theta) = C[exp(i theta Z_t)]_c = C[I cos(theta) + i Z_t sin(theta)]_c, C1eZ_{tc}(theta), CeZ1_{tc}(theta), or C1eZ1_{tc}(theta)
        // CeZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + e^{i theta} a_{10} |10> + e^{-i theta} |11>
        namespace exponential_pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex>
          struct exponential_pauli_cz_coeff_tcp
          {
            Complex const* phase_coefficient_ptr_;
            Complex const* conj_phase_coefficient_ptr_;

            exponential_pauli_cz_coeff_tcp(Complex const& phase_coefficient, Complex const& conj_phase_coefficient)
              : phase_coefficient_ptr_{std::addressof(phase_coefficient)},
                conj_phase_coefficient_ptr_{std::addressof(conj_phase_coefficient)}
            { }

            template <typename Iterator, typename StateInteger>
            void operator()(
              Iterator const, Iterator const, Iterator const first_10, Iterator const first_11, StateInteger const index, int const) const
            {
              *(first_10 + index) *= *phase_coefficient_ptr_;
              *(first_11 + index) *= *conj_phase_coefficient_ptr_;
            }
          }; // struct exponential_pauli_cz_coeff_tcp<Complex>

          template <typename Complex>
          inline ::ket::mpi::gate::page::exponential_pauli_z_detail::exponential_pauli_cz_coeff_tcp<Complex>
          make_exponential_pauli_cz_coeff_tcp(Complex const& phase_coefficient, Complex const& conj_phase_coefficient)
          { return {phase_coefficient, conj_phase_coefficient}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_cz_coeff_tcp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          assert(::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));
          assert(::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));

          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            [&phase_coefficient, &conj_phase_coefficient](
              auto const, auto const, auto const first_10, auto const first_11, StateInteger const index, int const)
            {
              *(first_10 + index) *= phase_coefficient;
              *(first_11 + index) *= conj_phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::two_page_qubits_gate<0u>(
            parallel_policy, local_state, permutated_target_qubit, permutated_control_qubit,
            ::ket::mpi::gate::page::exponential_pauli_z_detail::make_exponential_pauli_cz_coeff_tcp(phase_coefficient, conj_phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cz_coeff_tp: only target qubit is on page
        // CeZ_{tc}(theta) = C[exp(i theta Z_t)]_c = C[I cos(theta) + i Z_t sin(theta)]_c, C1eZ_{tc}(theta), CeZ1_{tc}(theta), or C1eZ1_{tc}(theta)
        // CeZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + e^{i theta} a_{10} |10> + e^{-i theta} |11>
        namespace exponential_pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct exponential_pauli_cz_coeff_tp
          {
            Complex const* phase_coefficient_ptr_;
            Complex const* conj_phase_coefficient_ptr_;
            StateInteger control_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            exponential_pauli_cz_coeff_tp(
              Complex const& phase_coefficient, Complex const& conj_phase_coefficient,
              StateInteger const control_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask)
              : phase_coefficient_ptr_{std::addressof(phase_coefficient)},
                conj_phase_coefficient_ptr_{std::addressof(conj_phase_coefficient)},
                control_qubit_mask_{control_qubit_mask},
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
              *(zero_first + one_index) *= *phase_coefficient_ptr_;
              *(one_first + one_index) *= *conj_phase_coefficient_ptr_;
            }
          }; // struct exponential_pauli_cz_coeff_tp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::exponential_pauli_z_detail::exponential_pauli_cz_coeff_tp<Complex, StateInteger>
          make_exponential_pauli_cz_coeff_tp(
            Complex const& phase_coefficient, Complex const& conj_phase_coefficient,
            StateInteger const control_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient, conj_phase_coefficient, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_cz_coeff_tp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

          assert(not ::ket::mpi::page::is_on_page(permutated_control_qubit, local_state));
          auto const control_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_control_qubit);
          auto const nonpage_lower_bits_mask = control_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            [&phase_coefficient, &conj_phase_coefficient, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const zero_first, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor control_qubit_mask;
              *(zero_first + one_index) *= phase_coefficient;
              *(one_first + one_index) *= conj_phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_target_qubit,
            ::ket::mpi::gate::page::exponential_pauli_z_detail::make_exponential_pauli_cz_coeff_tp(
              phase_coefficient, conj_phase_coefficient, control_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }

        // cz_coeff_cp: only control qubit is on page
        // CeZ_{tc}(theta) = C[exp(i theta Z_t)]_c = C[I cos(theta) + i Z_t sin(theta)]_c, C1eZ_{tc}(theta), CeZ1_{tc}(theta), or C1eZ1_{tc}(theta)
        // CeZ_{1,2}(theta) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
        //   = a_{00} |00> + a_{01} |01> + e^{i theta} a_{10} |10> + e^{-i theta} |11>
        namespace exponential_pauli_z_detail
        {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
          template <typename Complex, typename StateInteger>
          struct exponential_pauli_cz_coeff_cp
          {
            Complex const* phase_coefficient_ptr_;
            Complex const* conj_phase_coefficient_ptr_;
            StateInteger target_qubit_mask_;
            StateInteger nonpage_lower_bits_mask_;
            StateInteger nonpage_upper_bits_mask_;

            exponential_pauli_cz_coeff_cp(
              Complex const& phase_coefficient, Complex const& conj_phase_coefficient,
              StateInteger const target_qubit_mask,
              StateInteger const nonpage_lower_bits_mask,
              StateInteger const nonpage_upper_bits_mask) noexcept
              : phase_coefficient_ptr_{std::addressof(phase_coefficient)},
                conj_phase_coefficient_ptr_{std::addressof(conj_phase_coefficient)},
                target_qubit_mask_{target_qubit_mask},
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
              *(one_first + zero_index) *= *phase_coefficient_ptr_;
              *(one_first + one_index) *= *conj_phase_coefficient_ptr_;
            }
          }; // struct exponential_pauli_cz_coeff_cp<Complex, StateInteger>

          template <typename Complex, typename StateInteger>
          inline ::ket::mpi::gate::page::exponential_pauli_z_detail::exponential_pauli_cz_coeff_cp<Complex, StateInteger>
          make_exponential_pauli_cz_coeff_cp(
            Complex const& phase_coefficient, Complex const& conj_phase_coefficient,
            StateInteger const target_qubit_mask,
            StateInteger const nonpage_lower_bits_mask,
            StateInteger const nonpage_upper_bits_mask)
          { return {phase_coefficient, conj_phase_coefficient, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        } // namespace exponential_pauli_z_detail

        template <
          typename ParallelPolicy,
          typename RandomAccessRange, typename Complex, typename StateInteger, typename BitInteger>
        inline RandomAccessRange& exponential_pauli_cz_coeff_cp(
          ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state, Complex const& phase_coefficient,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_target_qubit,
          ::ket::mpi::permutated< ::ket::control< ::ket::qubit<StateInteger, BitInteger> > > const permutated_control_qubit)
        {
          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

          assert(not ::ket::mpi::page::is_on_page(permutated_target_qubit, local_state));

          auto const target_qubit_mask
            = ::ket::utility::integer_exp2<StateInteger>(permutated_target_qubit);
          auto const nonpage_lower_bits_mask = target_qubit_mask - StateInteger{1u};
          auto const nonpage_upper_bits_mask = compl nonpage_lower_bits_mask;

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            [&phase_coefficient, &conj_phase_coefficient, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask](
              auto const, auto const one_first, StateInteger const index_wo_nonpage_qubit, int const)
            {
              auto const zero_index
                = ((index_wo_nonpage_qubit bitand nonpage_upper_bits_mask) << 1u)
                  bitor (index_wo_nonpage_qubit bitand nonpage_lower_bits_mask);
              auto const one_index = zero_index bitor target_qubit_mask;
              *(one_first + zero_index) *= phase_coefficient;
              *(one_first + one_index) *= conj_phase_coefficient;
            });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          return ::ket::mpi::gate::page::detail::one_page_qubit_gate<1u>(
            parallel_policy, local_state, permutated_control_qubit,
            ::ket::mpi::gate::page::exponential_pauli_z_detail::make_exponential_pauli_cz_coeff_cp(
              phase_coefficient, conj_phase_coefficient, target_qubit_mask, nonpage_lower_bits_mask, nonpage_upper_bits_mask));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
        }
      } // namespace page
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_PAGE_EXPONENTIAL_PAULI_Z_STANDARD_HPP
