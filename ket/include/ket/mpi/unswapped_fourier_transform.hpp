#ifndef KET_MPI_UNSWAPPED_FOURIER_TRANSFORM_HPP
# define KET_MPI_UNSWAPPED_FOURIER_TRANSFORM_HPP

# include <boost/config.hpp>

# include <cassert>
# include <cstddef>
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_unsigned.hpp>
#   include <boost/utility/enable_if.hpp>
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif
# include <vector>

# include <boost/math/constants/constants.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/size.hpp>
# include <boost/range/value_type.hpp>
# ifdef BOOST_NO_CXX11_AUTO_DECLARATIONS
#   include <boost/range/iterator.hpp>
# endif

# include <yampi/environment.hpp>
# include <yampi/datatype.hpp>
# include <yampi/communicator.hpp>

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/generate_phase_coefficients.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/real_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/gate/hadamard.hpp>
# include <ket/mpi/gate/controlled_phase_shift.hpp>
# include <ket/mpi/utility/general_mpi.hpp>
# include <ket/mpi/utility/logger.hpp>
# include <ket/utility/integer_exp2.hpp>
# ifndef NDEBUG
#   include <ket/utility/is_unique.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
#   define KET_enable_if std::enable_if
# else
#   define KET_is_unsigned boost::is_unsigned
#   define KET_enable_if boost::enable_if_c
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif


namespace ket
{
  namespace mpi
  {
    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    unswapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      static_assert(
        KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(
        KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::range::is_unique(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print("Fourier", environment);

      std::size_t const num_qubits = boost::size(qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      typedef typename boost::range_iterator<Qubits const>::type qubits_iterator;
      qubits_iterator const qubits_first = boost::begin(qubits);

      for (std::size_t index = 0u; index < num_qubits; ++index)
      {
        std::size_t target_bit = num_qubits-index-1u;

        using ::ket::mpi::gate::hadamard;
        hadamard(
          mpi_policy, parallel_policy,
          local_state, qubits_first[target_bit], permutation,
          buffer, datatype, communicator, environment);

        for (std::size_t phase_exponent = 2u;
             phase_exponent <= num_qubits-index; ++phase_exponent)
        {
          std::size_t const control_bit = target_bit-(phase_exponent-1u);

          using ::ket::mpi::gate::controlled_phase_shift_coeff;
          controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficients[phase_exponent],
            qubits_first[target_bit], ::ket::make_control(qubits_first[control_bit]),
            permutation, buffer, datatype, communicator, environment);
        }
      }

      return local_state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    unswapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      return unswapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    unswapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      return unswapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    unswapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<RandomAccessRange>::type complex_type;
      std::vector<complex_type> phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return unswapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    unswapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<RandomAccessRange>::type complex_type;
      std::vector<complex_type> phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return unswapped_fourier_transform(
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    unswapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<RandomAccessRange>::type complex_type;
      std::vector<complex_type> phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return unswapped_fourier_transform(
        parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }



    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_unswapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      static_assert(
        KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
      static_assert(
        KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");
      assert(::ket::utility::range::is_unique(qubits));

      ::ket::mpi::utility::log_with_time_guard<char> print("Adj(Fourier)", environment);

      std::size_t const num_qubits = boost::size(qubits);
      ::ket::utility::generate_phase_coefficients(phase_coefficients, num_qubits);

      typedef typename boost::range_iterator<Qubits const>::type qubits_iterator;
      qubits_iterator const qubits_first = boost::begin(qubits);

      for (std::size_t target_bit = 0u; target_bit < num_qubits; ++target_bit)
      {
        for (std::size_t index = 0u; index < target_bit; ++index)
        {
          std::size_t const phase_exponent = 1u+target_bit-index;
          std::size_t const control_bit = target_bit-(phase_exponent-1u);

          using ::ket::mpi::gate::adj_controlled_phase_shift_coeff;
          adj_controlled_phase_shift_coeff(
            mpi_policy, parallel_policy,
            local_state, phase_coefficients[phase_exponent], qubits_first[target_bit],
            ::ket::make_control(qubits_first[control_bit]),
            permutation, buffer, datatype, communicator, environment);
        }

        using ::ket::mpi::gate::adj_hadamard;
        adj_hadamard(
          mpi_policy, parallel_policy,
          local_state, qubits_first[target_bit], permutation,
          buffer, datatype, communicator, environment);
      }

      return local_state;
    }

    template <
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_unswapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      return adj_unswapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits, typename PhaseCoefficientsAllocator,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_unswapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type,
        PhaseCoefficientsAllocator>& phase_coefficients,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      return adj_unswapped_fourier_transform(
        ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }


    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      RandomAccessRange&>::type
    adj_unswapped_fourier_transform(
      MpiPolicy const mpi_policy, ParallelPolicy const parallel_policy,
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<RandomAccessRange>::type complex_type;
      std::vector<complex_type> phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_unswapped_fourier_transform(
        mpi_policy, parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<RandomAccessRange>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<RandomAccessRange>::value),
      RandomAccessRange&>::type
    adj_unswapped_fourier_transform(
      RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<RandomAccessRange>::type complex_type;
      std::vector<complex_type> phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_unswapped_fourier_transform(
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }

    template <
      typename ParallelPolicy,
      typename RandomAccessRange, typename Qubits,
      typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      RandomAccessRange&>::type
    adj_unswapped_fourier_transform(
      ParallelPolicy const parallel_policy, RandomAccessRange& local_state,
      Qubits const& qubits,
      ::ket::mpi::qubit_permutation<
        StateInteger, BitInteger, Allocator>& permutation,
      std::vector<
        typename boost::range_value<RandomAccessRange>::type, BufferAllocator>& buffer,
      yampi::datatype const datatype,
      yampi::communicator const communicator,
      yampi::environment const& environment)
    {
      typedef typename boost::range_value<RandomAccessRange>::type complex_type;
      std::vector<complex_type> phase_coefficients
        = ::ket::utility::generate_phase_coefficients<complex_type>(boost::size(qubits));

      return adj_unswapped_fourier_transform(
        parallel_policy,
        local_state, qubits, phase_coefficients, permutation,
        buffer, datatype, communicator, environment);
    }
  } // namespace mpi
} // namespace ket


# undef KET_enable_if
# undef KET_is_unsigned
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

