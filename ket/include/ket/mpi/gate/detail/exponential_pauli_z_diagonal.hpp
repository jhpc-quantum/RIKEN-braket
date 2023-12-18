#ifndef KET_MPI_GATE_DETAIL_EXPONENTIAL_PAULI_Z_DIAGONAL_HPP
# define KET_MPI_GATE_DETAIL_EXPONENTIAL_PAULI_Z_DIAGONAL_HPP

# include <boost/config.hpp>

# include <complex>
# include <vector>
# include <array>

# include <boost/range/value_type.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# ifdef KET_PRINT_LOG
#   include <ket/qubit_io.hpp>
# endif // KET_PRINT_LOG
# include <ket/gate/exponential_pauli_z.hpp>
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
#   include <ket/mpi/permutated.hpp>
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/mpi/gate/page/exponential_pauli_z.hpp>
# include <ket/mpi/page/is_on_page.hpp>


namespace ket
{
  namespace mpi
  {
    namespace gate
    {
      // exponential_pauli_z_coeff
      // eZ_i(s) = exp(is Z_i) = I cos s + i Z_i sin s
      // eZ_1(s) (a_0 |0> + a_1 |1>) = e^{is} a_0 |0> + e^{-is} a_1 |1>
      // eZZ_{ij}(s) = exp(is Z_i Z_j) = I cos s + i Z_i Z_j sin s
      // eZZ_{1,2}(s) (a_{00} |00> + a_{01} |01> + a_{10} |10> + a{11} |11>)
      //   = e^{is} a_{00} |00> + e^{-is} a_{01} |01> + e^{-is} a_{10} |10> + e^{is} a_{11} |11>
      namespace exponential_pauli_z_detail
      {
# ifdef BOOST_NO_CXX14_GENERIC_LAMBDAS
        template <typename Complex, typename StateInteger>
        struct multiplication_assignment
        {
         private:
          Complex scalar_;

         public:
          multiplication_assignment(Complex const& scalar)
            : scalar_{scalar}
          { }

          template <typename Iterator>
          void operator()(Iterator const iter, StateInteger const) const
          { *iter *= scalar_; }
        }; // struct multiplication_assignment<Complex, StateInteger>

        template <typename StateInteger, typename Complex>
        inline ::ket::mpi::gate::exponential_pauli_z_detail::multiplication_assignment<Complex, StateInteger> make_multiplication_assignment(Complex const& scalar)
        { return {scalar}; }
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& exponential_pauli_z_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const qubit,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit = permutation[qubit];
          if (::ket::mpi::page::is_on_page(permutated_qubit, local_state))
            return ::ket::mpi::gate::page::exponential_pauli_z_coeff1(
              parallel_policy, local_state, phase_coefficient, permutated_qubit);

          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; },
            [&conj_phase_coefficient](auto const iter, StateInteger const) { *iter *= conj_phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit,
            ::ket::mpi::gate::exponential_pauli_z_detail::make_multiplication_assignment<StateInteger>(phase_coefficient),
            ::ket::mpi::gate::exponential_pauli_z_detail::make_multiplication_assignment<StateInteger>(conj_phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          return local_state;
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange,
          typename StateInteger, typename BitInteger, typename Allocator>
        inline RandomAccessRange& exponential_pauli_z_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const permutated_qubit1 = permutation[qubit1];
          auto const permutated_qubit2 = permutation[qubit2];
          if (::ket::mpi::page::is_on_page(permutated_qubit1, local_state))
          {
            if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
              return ::ket::mpi::gate::page::exponential_pauli_z_coeff2_2p(
                parallel_policy, local_state, phase_coefficient, permutated_qubit1, permutated_qubit2);

            return ::ket::mpi::gate::page::exponential_pauli_z_coeff2_p(
              parallel_policy, local_state, phase_coefficient, permutated_qubit1, permutated_qubit2);
          }
          else if (::ket::mpi::page::is_on_page(permutated_qubit2, local_state))
            return ::ket::mpi::gate::page::exponential_pauli_z_coeff2_p(
              parallel_policy, local_state, phase_coefficient, permutated_qubit2, permutated_qubit1);

          using std::conj;
          auto const conj_phase_coefficient = conj(phase_coefficient);

# ifndef BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit1, qubit2,
            [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; },
            [&conj_phase_coefficient](auto const iter, StateInteger const) { *iter *= conj_phase_coefficient; },
            [&conj_phase_coefficient](auto const iter, StateInteger const) { *iter *= conj_phase_coefficient; },
            [&phase_coefficient](auto const iter, StateInteger const) { *iter *= phase_coefficient; });
# else // BOOST_NO_CXX14_GENERIC_LAMBDAS
          ::ket::mpi::utility::diagonal_loop(
            mpi_policy, parallel_policy,
            local_state, permutation, communicator, environment, qubit1, qubit2,
            ::ket::mpi::gate::exponential_pauli_z_detail::make_multiplication_assignment<StateInteger>(phase_coefficient),
            ::ket::mpi::gate::exponential_pauli_z_detail::make_multiplication_assignment<StateInteger>(conj_phase_coefficient),
            ::ket::mpi::gate::exponential_pauli_z_detail::make_multiplication_assignment<StateInteger>(conj_phase_coefficient),
            ::ket::mpi::gate::exponential_pauli_z_detail::make_multiplication_assignment<StateInteger>(phase_coefficient));
# endif // BOOST_NO_CXX14_GENERIC_LAMBDAS

          return local_state;
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename... Qubits, typename Allocator>
        inline RandomAccessRange& do_exponential_pauli_z_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2,
          ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, communicator, environment);
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, communicator, environment);

          auto const first = std::begin(local_state);
          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
            ::ket::gate::exponential_pauli_z_coeff(
              parallel_policy,
              first + data_block_index * data_block_size,
              first + (data_block_index + 1u) * data_block_size,
              permutation[qubit1], permutation[qubit2], permutation[qubit3], permutation[qubits]...);

          return local_state;
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& exponential_pauli_z_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2,
          ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(Qubits) + 1u>{qubit1, qubit2, qubit3, qubits...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, communicator, environment);

          return ::ket::mpi::gate::exponential_pauli_z_detail::do_exponential_pauli_z_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, qubit1, qubit2, qubit3, qubits..., permutation, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& exponential_pauli_z_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const qubit1,
          ::ket::qubit<StateInteger, BitInteger> const qubit2,
          ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
          auto qubit_array = std::array<qubit_type, sizeof...(Qubits) + 3u>{qubit1, qubit2, qubit3, qubits...};
          ::ket::mpi::utility::maybe_interchange_qubits(
            mpi_policy, parallel_policy,
            local_state, qubit_array, permutation, buffer, datatype, communicator, environment);

          return ::ket::mpi::gate::exponential_pauli_z_detail::do_exponential_pauli_z_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficient, qubit1, qubit2, qubit3, qubits..., permutation, communicator, environment);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZ(coeff) "}, phase_coefficient, ' ', qubit), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZ(coeff) "}, phase_coefficient, ' ', qubit), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZZ(coeff) "}, phase_coefficient, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZZ(coeff) "}, phase_coefficient, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace exponential_pauli_z_detail
      {
        inline std::string do_generate_exponential_pauli_z_coeff_string(std::string const& result)
        { return result; }

        template <typename StateInteger, typename BitInteger, typename... Qubits>
        inline std::string do_generate_exponential_pauli_z_coeff_string(std::string const& result, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        { return do_generate_exponential_pauli_z_coeff_string(::ket::mpi::utility::generate_logger_string(result, ' ', qubit), qubits...); }

        template <typename Complex, typename... Qubits>
        inline std::string generate_exponential_pauli_z_coeff_string(Complex const& phase_coefficient, Qubits const... qubits)
        {
          auto result = std::string{"e"};
          for (std::size_t count = 0u; count < sizeof...(Qubits); ++count)
            result += 'Z';
          result += "(coeff) ";
          result = ::ket::mpi::utility::generate_logger_string(result, phase_coefficient);

          return do_generate_exponential_pauli_z_coeff_string(result, qubits...);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_exponential_pauli_z_coeff_string(phase_coefficient, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_exponential_pauli_z_coeff_string(phase_coefficient, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, datatype, communicator, environment);
      }

      namespace exponential_pauli_z_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_exponential_pauli_z_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using std::conj;
          return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
            mpi_policy, parallel_policy,
            local_state, conj(phase_coefficient), qubit, qubits..., permutation,
            buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Complex,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_exponential_pauli_z_coeff(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using std::conj;
          return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
            mpi_policy, parallel_policy,
            local_state, conj(phase_coefficient), qubit, qubits..., permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZ(coeff)) "}, phase_coefficient, ' ', qubit), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZ(coeff)) "}, phase_coefficient, ' ', qubit1), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZZ(coeff)) "}, phase_coefficient, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZZ(coeff)) "}, phase_coefficient, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace exponential_pauli_z_detail
      {
        inline std::string do_generate_adj_exponential_pauli_z_coeff_string(std::string const& result)
        { return result; }

        template <typename StateInteger, typename BitInteger, typename... Qubits>
        inline std::string do_generate_adj_exponential_pauli_z_coeff_string(std::string const& result, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        { return do_generate_adj_exponential_pauli_z_coeff_string(::ket::mpi::utility::generate_logger_string(result, ' ', qubit), qubits...); }

        template <typename Complex, typename... Qubits>
        inline std::string generate_adj_exponential_pauli_z_coeff_string(Complex const& phase_coefficient, Qubits const... qubits)
        {
          auto result = std::string{"Adj(e"};
          for (std::size_t count = 0u; count < sizeof...(Qubits); ++count)
            result += 'Z';
          result += "(coeff)) ";
          result = ::ket::mpi::utility::generate_logger_string(result, phase_coefficient);

          return do_generate_adj_exponential_pauli_z_coeff_string(result, qubits...);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_adj_exponential_pauli_z_coeff_string(phase_coefficient, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_adj_exponential_pauli_z_coeff_string(phase_coefficient, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z_coeff(
          mpi_policy, parallel_policy,
          local_state, phase_coefficient, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Complex,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z_coeff(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Complex const& phase_coefficient, // exp(i theta) = cos(theta) + i sin(theta)
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z_coeff(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase_coefficient, qubit, qubits..., permutation,
          buffer, datatype, communicator, environment);
      }

      // exponential_pauli_z
      namespace exponential_pauli_z_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& exponential_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase), qubit, qubits..., permutation,
            buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& exponential_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          using complex_type = typename boost::range_value<RandomAccessRange>::type;
          return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z_coeff(
            mpi_policy, parallel_policy,
            local_state, ::ket::utility::exp_i<complex_type>(phase), qubit, qubits..., permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZ "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZ "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZZ "}, phase, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, permutation,
          buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"eZZ "}, phase, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, permutation,
          buffer, datatype, communicator, environment);
      }

      namespace exponential_pauli_z_detail
      {
        inline std::string do_generate_exponential_pauli_z_string(std::string const& result)
        { return result; }

        template <typename StateInteger, typename BitInteger, typename... Qubits>
        inline std::string do_generate_exponential_pauli_z_string(std::string const& result, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        { return do_generate_exponential_pauli_z_string(::ket::mpi::utility::generate_logger_string(result, ' ', qubit), qubits...); }

        template <typename Real, typename... Qubits>
        inline std::string generate_exponential_pauli_z_string(Real const phase, Qubits const... qubits)
        {
          auto result = std::string{"e"};
          for (std::size_t count = 0u; count < sizeof...(Qubits); ++count)
            result += 'Z';
          result = ::ket::mpi::utility::generate_logger_string(result, ' ', phase_coefficient);

          return do_generate_exponential_pauli_z_string(result, qubits...);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_exponential_pauli_z_string(phase, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_exponential_pauli_z_string(phase, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, qubits..., permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, qubits..., permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& exponential_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, qubits..., permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& exponential_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, qubits..., permutation, buffer, datatype, communicator, environment);
      }

      namespace exponential_pauli_z_detail
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator>
        inline RandomAccessRange& adj_exponential_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
            mpi_policy, parallel_policy,
            local_state, -phase, qubit, qubits..., permutation,
            buffer, communicator, environment);
        }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename RandomAccessRange, typename Real,
          typename StateInteger, typename BitInteger, typename... Qubits,
          typename Allocator, typename BufferAllocator, typename DerivedDatatype>
        inline RandomAccessRange& adj_exponential_pauli_z(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          RandomAccessRange& local_state,
          Real const phase,
          ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::communicator const& communicator, yampi::environment const& environment)
        {
          return ::ket::mpi::gate::exponential_pauli_z_detail::exponential_pauli_z(
            mpi_policy, parallel_policy,
            local_state, -phase, qubit, qubits..., permutation,
            buffer, datatype, communicator, environment);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZ) "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZ) "}, phase, ' ', qubit), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZZ) "}, phase, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Adj(eZZ) "}, phase, ' ', qubit1, ' ', qubit2), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, permutation, buffer, datatype, communicator, environment);
      }

      namespace exponential_pauli_z_detail
      {
        inline std::string do_generate_adj_exponential_pauli_z_string(std::string const& result)
        { return result; }

        template <typename StateInteger, typename BitInteger, typename... Qubits>
        inline std::string do_generate_adj_exponential_pauli_z_string(std::string const& result, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
        { return do_generate_adj_exponential_pauli_z_string(::ket::mpi::utility::generate_logger_string(result, ' ', qubit), qubits...); }

        template <typename Real, typename... Qubits>
        inline std::string generate_adj_exponential_pauli_z_string(Real const phase, Qubits const... qubits)
        {
          auto result = std::string{"Adj(e"};
          for (std::size_t count = 0u; count < sizeof...(Qubits); ++count)
            result += 'Z';
          result = ::ket::mpi::utility::generate_logger_string(result, ") ", phase_coefficient);

          return do_generate_adj_exponential_pauli_z_string(result, qubits...);
        }
      } // namespace exponential_pauli_z_detail

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_adj_exponential_pauli_z_string(phase, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit1,
        ::ket::qubit<StateInteger, BitInteger> const qubit2,
        ::ket::qubit<StateInteger, BitInteger> const qubit3, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::gate::exponential_pauli_z_detail::generate_adj_exponential_pauli_z_string(phase, qubit1, qubit2, qubit3, qubits...), environment};

        return ::ket::mpi::gate::exponential_pauli_z_detail::adj_exponential_pauli_z(
          mpi_policy, parallel_policy,
          local_state, phase, qubit1, qubit2, qubit3, qubits...,
          permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, qubits..., permutation, buffer, communicator, environment);
      }

      template <
        typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z(
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, phase, qubit, qubits..., permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator>
      inline RandomAccessRange& adj_exponential_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, qubits..., permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename RandomAccessRange, typename Real,
        typename StateInteger, typename BitInteger, typename... Qubits,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      inline RandomAccessRange& adj_exponential_pauli_z(
        ParallelPolicy const parallel_policy,
        RandomAccessRange& local_state,
        Real const phase,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      {
        return ::ket::mpi::gate::adj_exponential_pauli_z(
          ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
          local_state, phase, qubit, qubits..., permutation, buffer, datatype, communicator, environment);
      }
    } // namespace gate
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_GATE_DETAIL_EXPONENTIAL_PAULI_Z_DIAGONAL_HPP
