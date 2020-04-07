#ifndef KET_MPI_GENERAL_MPI_HPP
# define KET_MPI_GENERAL_MPI_HPP

# include <boost/config.hpp>

# include <cstddef>
# include <cassert>
# include <iostream>
# include <vector>
# include <algorithm>
# include <iterator>
# include <utility>
# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   include <array>
# else
#   include <boost/array.hpp>
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/is_unsigned.hpp>
#   include <boost/type_traits/is_same.hpp>
#   include <boost/type_traits/integral_constant.hpp>
# endif
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   include <boost/static_assert.hpp>
# endif

# ifndef NDEBUG
#   include <boost/optional.hpp>
# endif

# include <boost/range/value_type.hpp>
# include <boost/range/size.hpp>
# include <boost/range/empty.hpp>
# include <boost/range/numeric.hpp>

# ifdef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#   include <boost/preprocessor/arithmetic/dec.hpp>
#   include <boost/preprocessor/repetition/repeat.hpp>
#   include <boost/preprocessor/repetition/enum_trailing_params.hpp>
#   include <boost/preprocessor/repetition/enum_trailing_binary_params.hpp>
#   ifndef KET_MAX_NUM_CONTROL_QUBITS
#     define KET_MAX_NUM_CONTROL_QUBITS 2
#   endif
# endif

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>
# ifndef NDEBUG
#   include <yampi/lowest_io_process.hpp>
# endif

# include <ket/qubit.hpp>
# include <ket/control.hpp>
# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>
# include <ket/utility/loop_n.hpp>
# include <ket/utility/begin.hpp>
# include <ket/utility/end.hpp>
# include <ket/utility/meta/const_iterator_of.hpp>
# include <ket/mpi/qubit_permutation.hpp>
# include <ket/mpi/utility/detail/swap_local_qubits.hpp>
# include <ket/mpi/utility/detail/interchange_qubits.hpp>
# include <ket/mpi/utility/logger.hpp>
# ifndef NDEBUG
#   include <ket/qubit_io.hpp>
#   include <ket/mpi/qubit_permutation_io.hpp>
# endif

# ifndef BOOST_NO_CXX11_HDR_ARRAY
#   define KET_array std::array
# else
#   define KET_array boost::array
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_is_unsigned std::is_unsigned
#   define KET_is_same std::is_same
#   define KET_true_type std::true_type
#   define KET_false_type std::false_type
# else
#   define KET_is_unsigned boost::is_unsigned
#   define KET_is_same boost::is_same
#   define KET_true_type boost::true_type
#   define KET_false_type boost::false_type
# endif

# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
# endif

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T&&
#   define KET_FORWARD_OR_COPY(T, x) std::forward<T>(x)
# else
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T
#   define KET_FORWARD_OR_COPY(T, x) x
# endif


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace policy
      {
        class general_mpi
        {
         public:
          explicit general_mpi() BOOST_NOEXCEPT_OR_NOTHROW
          { }
        };

        inline general_mpi make_general_mpi() BOOST_NOEXCEPT_OR_NOTHROW
        { return general_mpi();  }

        namespace meta
        {
          template <typename T>
          struct is_mpi_policy
            : KET_false_type
          { };

          template <>
          struct is_mpi_policy< ::ket::mpi::utility::policy::general_mpi >
            : KET_true_type
          { };
        } // namespace meta
      } // namespace policy


      namespace general_mpi_detail
      {
        template <
          typename ParallelPolicy, typename LocalState,
          typename StateInteger, typename BitInteger, typename Allocator,
          typename UnswappableQubits>
        inline ::ket::qubit<StateInteger, BitInteger>
        make_local_swap_qubit_swappable(
          ParallelPolicy const parallel_policy,
          LocalState& local_state,
          ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
          UnswappableQubits const& unswappable_qubits,
          ::ket::qubit<StateInteger, BitInteger> const permutated_local_swap_qubit)
        {
          typedef ket::qubit<StateInteger, BitInteger> qubit_type;
          static_assert(
            (KET_is_same<
               typename boost::range_value<UnswappableQubits>::type,
               qubit_type>::value),
            "value_type of UnswappableQubits must be the same to qubit_type");

          using ::ket::mpi::inverse;
          qubit_type const local_swap_qubit
            = inverse(permutation)[permutated_local_swap_qubit];

          typename ::ket::utility::meta::const_iterator_of<UnswappableQubits const>::type const last
            = ::ket::utility::end(unswappable_qubits);

          if (std::find(
                ::ket::utility::begin(unswappable_qubits), last, local_swap_qubit)
              != last)
          {
            qubit_type permutated_other_qubit = permutated_local_swap_qubit;
            qubit_type other_qubit;
            do
            {
              --permutated_other_qubit;
              using ::ket::mpi::inverse;
              other_qubit
                = inverse(permutation)[permutated_other_qubit];
            }
            while (std::find(
                     ::ket::utility::begin(unswappable_qubits), last, other_qubit)
                   != last);

            ::ket::mpi::utility::detail::swap_local_qubits(
              parallel_policy, local_state,
              permutated_local_swap_qubit, permutated_other_qubit);
            using ::ket::mpi::permutate;
            permutate(permutation, local_swap_qubit, other_qubit);
          }

          return inverse(permutation)[permutated_local_swap_qubit];
        }
      } // namespace general_mpi_detail


      namespace dispatch
      {
        // TODO: Modify maybe_interchange_qubits. This implementation of maybe_interchange_qubits has redundancy
        template <std::size_t num_qubits_of_operation, typename MpiPolicy>
        struct maybe_interchange_qubits;

        template <>
        struct maybe_interchange_qubits<
          1u, ::ket::mpi::utility::policy::general_mpi>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, 1u> const& qubits,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits> const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));


            typedef ket::qubit<StateInteger, BitInteger> qubit_type;

            BitInteger const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));
            qubit_type const permutated_global_swap_qubit
              = permutation[qubits[0u]];

            if (static_cast<BitInteger>(permutated_global_swap_qubit)
                < num_local_qubits)
              return;


            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<1>", environment);

# ifndef NDEBUG
            boost::optional<yampi::rank> const maybe_io_rank = yampi::lowest_io_process(environment);
            yampi::rank const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            //  Swaps between xxxbxxx|(~b)xxxxxxxxx and xxx(~b)xxx|bxxxxxxxxx.
            // Upper qubits are global qubits representing MPI rank. Lower
            // qubits are local qubits representing memory address. The first
            // upper qubit in the local qubits is a "local swap qubit". A bit in
            // global qubits and the "local swap qubit" would be swapped. If the
            // first upper qubit in the local qubits is an unswappable qubit, it
            // and a lowerer (swappable) qubit should be swapped before this
            // process.

            qubit_type const permutated_local_swap_qubit(num_local_qubits-1u);
            qubit_type const local_swap_qubit
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG


            // xxxxxbxx(|xxxxxxxxxx)
            StateInteger const source_global_index
              = static_cast<StateInteger>(communicator.rank(environment).mpi_rank());
            // xxxxx(~b)xx(|xxxxxxxxxx)
            StateInteger const target_global_index
              = source_global_index
                xor ((static_cast<StateInteger>(1u) << permutated_global_swap_qubit)
                     >> num_local_qubits);

            // (00000000|)(~b)0000000000
            StateInteger const source_local_first_index
              = ((target_global_index << num_local_qubits)
                 bitand ket::utility::integer_exp2<StateInteger>(
                          permutated_global_swap_qubit))
                >> (permutated_global_swap_qubit-permutated_local_swap_qubit);
            // (00000000|)0111111111
            StateInteger const prev_last_mask
              = (static_cast<StateInteger>(1u) << permutated_local_swap_qubit)
                - static_cast<StateInteger>(1u);
            // (00000000|)(~b)1111111111 + 1
            StateInteger const source_local_last_index
              = (source_local_first_index bitor prev_last_mask)
                + static_cast<StateInteger>(1u);

            {
            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<1>::swap", environment);

            ::ket::mpi::utility::detail::interchange_qubits(
              local_state, buffer, source_local_first_index, source_local_last_index,
              static_cast<yampi::rank>(target_global_index), communicator, environment);
            }

            using ::ket::mpi::permutate;
            permutate(permutation, qubits[0u], local_swap_qubit);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, 1u> const& qubits,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits> const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));


            typedef ket::qubit<StateInteger, BitInteger> qubit_type;

            BitInteger const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));
            qubit_type const permutated_global_swap_qubit
              = permutation[qubits[0u]];

            if (static_cast<BitInteger>(permutated_global_swap_qubit)
                < num_local_qubits)
              return;


            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<1>", environment);

# ifndef NDEBUG
            boost::optional<yampi::rank> const maybe_io_rank = yampi::lowest_io_process(environment);
            yampi::rank const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            //  Swaps between xxxbxxx|(~b)xxxxxxxxx and xxx(~b)xxx|bxxxxxxxxx.
            // Upper qubits are global qubits representing MPI rank. Lower
            // qubits are local qubits representing memory address. The first
            // upper qubit in the local qubits is a "local swap qubit". A bit in
            // global qubits and the "local swap qubit" would be swapped. If the
            // first upper qubit in the local qubits is an unswappable qubit, it
            // and a lowerer (swappable) qubit should be swapped before this
            // process.

            qubit_type const permutated_local_swap_qubit(num_local_qubits-1u);
            qubit_type const local_swap_qubit
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG


            // xxxxxbxx(|xxxxxxxxxx)
            StateInteger const source_global_index
              = static_cast<StateInteger>(communicator.rank(environment).mpi_rank());
            // xxxxx(~b)xx(|xxxxxxxxxx)
            StateInteger const target_global_index
              = source_global_index
                xor ((static_cast<StateInteger>(1u) << permutated_global_swap_qubit)
                     >> num_local_qubits);

            // (00000000|)(~b)0000000000
            StateInteger const source_local_first_index
              = ((target_global_index << num_local_qubits)
                 bitand ket::utility::integer_exp2<StateInteger>(
                          permutated_global_swap_qubit))
                >> (permutated_global_swap_qubit-permutated_local_swap_qubit);
            // (00000000|)0111111111
            StateInteger const prev_last_mask
              = (static_cast<StateInteger>(1u) << permutated_local_swap_qubit)
                - static_cast<StateInteger>(1u);
            // (00000000|)(~b)1111111111 + 1
            StateInteger const source_local_last_index
              = (source_local_first_index bitor prev_last_mask)
                + static_cast<StateInteger>(1u);

            {
            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<1>::swap", environment);

            ::ket::mpi::utility::detail::interchange_qubits(
              local_state, buffer, source_local_first_index, source_local_last_index,
              datatype, static_cast<yampi::rank>(target_global_index),
              communicator, environment);
            }

            using ::ket::mpi::permutate;
            permutate(permutation, qubits[0u], local_swap_qubit);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }
        };

        template <>
        struct maybe_interchange_qubits<
          2u, ::ket::mpi::utility::policy::general_mpi>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, 2u> const& qubits,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits> const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));


            typedef ket::qubit<StateInteger, BitInteger> qubit_type;

            BitInteger const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));
            qubit_type const permutated_global_swap_qubit0
              = permutation[qubits[0u]];
            qubit_type const permutated_global_swap_qubit1
              = permutation[qubits[1u]];


            if (static_cast<BitInteger>(permutated_global_swap_qubit0)
                < num_local_qubits)
            {
              KET_array<qubit_type, 1u> new_qubits = { qubits[1u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[0u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  1u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_single_qubit;

              maybe_interchange_single_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, communicator, environment);

              return;
            }
            else if (static_cast<BitInteger>(permutated_global_swap_qubit1)
                  < num_local_qubits)
            {
              KET_array<qubit_type, 1u> new_qubits = { qubits[0u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[1u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  1u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_single_qubit;

              maybe_interchange_single_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, communicator, environment);

              return;
            }


            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<2>", environment);

# ifndef NDEBUG
            boost::optional<yampi::rank> const maybe_io_rank = yampi::lowest_io_process(environment);
            yampi::rank const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            //  Swaps between xxbxb'xx|cc'xxxxxxxx and
            // xxcxc'xx|bb'xxxxxxxx (c = b or ~b). Upper qubits are global
            // qubits representing MPI rank. Lower qubits are local qubits
            // representing memory address. The first two upper qubits in the
            // local qubits are "local swap qubits". Two consecutive bits in
            // global qubits and the "local swap qubits" would be swapped.

            qubit_type const permutated_local_swap_qubit0(num_local_qubits-1u);
            qubit_type const local_swap_qubit0
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit0);
            qubit_type const permutated_local_swap_qubit1(num_local_qubits-2u);
            qubit_type const local_swap_qubit1
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit1);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG

            // xxbxb'xx(|xxxxxxxxxx)
            StateInteger const source_global_index
              = static_cast<StateInteger>(communicator.rank(environment).mpi_rank());

            for (StateInteger target_global_mask = 1u;
                 target_global_mask < ::ket::utility::integer_exp2<StateInteger>(2u);
                 ++target_global_mask)
            {
              StateInteger const target_global_mask0
                = target_global_mask bitand static_cast<StateInteger>(1u);
              StateInteger const target_global_mask1
                = (target_global_mask bitand static_cast<StateInteger>(2u)) >> 1;

              // xxcxc'xx(|xxxxxxxxxx) (c = b or b')
              StateInteger const target_global_index
                = source_global_index
                  xor
                  (((target_global_mask0 << permutated_global_swap_qubit0)
                    >> num_local_qubits)
                   bitor
                   ((target_global_mask1 << permutated_global_swap_qubit1)
                    >> num_local_qubits));

              // (0000000|)c0000000000
              StateInteger const source_local_first_index0
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit0))
                  >> (permutated_global_swap_qubit0-permutated_local_swap_qubit0);
              // (0000000|)0c'000000000
              StateInteger const source_local_first_index1
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit1))
                  >> (permutated_global_swap_qubit1-permutated_local_swap_qubit1);
              // (0000000|)cc'000000000
              StateInteger const source_local_first_index
                = source_local_first_index0 bitor source_local_first_index1;
              // (0000000|)0011111111
              StateInteger const prev_last_mask
                = (static_cast<StateInteger>(1u) << permutated_local_swap_qubit1)
                  - static_cast<StateInteger>(1u);
              // (0000000|)cc'111111111 + 1
              StateInteger const source_local_last_index
                = (source_local_first_index bitor prev_last_mask)
                  + static_cast<StateInteger>(1u);

              ::ket::mpi::utility::log_with_time_guard<char> print(
                "interchange_qubits<2>::swap", environment);

              ::ket::mpi::utility::detail::interchange_qubits(
                local_state, buffer, source_local_first_index, source_local_last_index,
                static_cast<yampi::rank>(target_global_index),
                communicator, environment);
            }

            using ::ket::mpi::permutate;
            permutate(permutation, qubits[0u], local_swap_qubit0);
            permutate(permutation, qubits[1u], local_swap_qubit1);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, 2u> const& qubits,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits> const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));


            typedef ket::qubit<StateInteger, BitInteger> qubit_type;

            BitInteger const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));
            qubit_type const permutated_global_swap_qubit0
              = permutation[qubits[0u]];
            qubit_type const permutated_global_swap_qubit1
              = permutation[qubits[1u]];


            if (static_cast<BitInteger>(permutated_global_swap_qubit0)
                < num_local_qubits)
            {
              KET_array<qubit_type, 1u> new_qubits = { qubits[1u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[0u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  1u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_single_qubit;

              maybe_interchange_single_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);

              return;
            }
            else if (static_cast<BitInteger>(permutated_global_swap_qubit1)
                  < num_local_qubits)
            {
              KET_array<qubit_type, 1u> new_qubits = { qubits[0u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[1u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  1u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_single_qubit;

              maybe_interchange_single_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);

              return;
            }


            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<2>", environment);

# ifndef NDEBUG
            boost::optional<yampi::rank> const maybe_io_rank = yampi::lowest_io_process(environment);
            yampi::rank const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            //  Swaps between xxbxb'xx|cc'xxxxxxxx and
            // xxcxc'xx|bb'xxxxxxxx (c = b or ~b). Upper qubits are global
            // qubits representing MPI rank. Lower qubits are local qubits
            // representing memory address. The first two upper qubits in the
            // local qubits are "local swap qubits". Two consecutive bits in
            // global qubits and the "local swap qubits" would be swapped.

            qubit_type const permutated_local_swap_qubit0(num_local_qubits-1u);
            qubit_type const local_swap_qubit0
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit0);
            qubit_type const permutated_local_swap_qubit1(num_local_qubits-2u);
            qubit_type const local_swap_qubit1
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit1);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG

            // xxbxb'xx(|xxxxxxxxxx)
            StateInteger const source_global_index
              = static_cast<StateInteger>(communicator.rank(environment).mpi_rank());

            for (StateInteger target_global_mask = 1u;
                 target_global_mask < ::ket::utility::integer_exp2<StateInteger>(2u);
                 ++target_global_mask)
            {
              StateInteger const target_global_mask0
                = target_global_mask bitand static_cast<StateInteger>(1u);
              StateInteger const target_global_mask1
                = (target_global_mask bitand static_cast<StateInteger>(2u)) >> 1;

              // xxcxc'xx(|xxxxxxxxxx) (c = b or b')
              StateInteger const target_global_index
                = source_global_index
                  xor
                  (((target_global_mask0 << permutated_global_swap_qubit0)
                    >> num_local_qubits)
                   bitor
                   ((target_global_mask1 << permutated_global_swap_qubit1)
                    >> num_local_qubits));

              // (0000000|)c0000000000
              StateInteger const source_local_first_index0
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit0))
                  >> (permutated_global_swap_qubit0-permutated_local_swap_qubit0);
              // (0000000|)0c'000000000
              StateInteger const source_local_first_index1
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit1))
                  >> (permutated_global_swap_qubit1-permutated_local_swap_qubit1);
              // (0000000|)cc'000000000
              StateInteger const source_local_first_index
                = source_local_first_index0 bitor source_local_first_index1;
              // (0000000|)0011111111
              StateInteger const prev_last_mask
                = (static_cast<StateInteger>(1u) << permutated_local_swap_qubit1)
                  - static_cast<StateInteger>(1u);
              // (0000000|)cc'111111111 + 1
              StateInteger const source_local_last_index
                = (source_local_first_index bitor prev_last_mask)
                  + static_cast<StateInteger>(1u);

              ::ket::mpi::utility::log_with_time_guard<char> print(
                "interchange_qubits<2>::swap", environment);

              ::ket::mpi::utility::detail::interchange_qubits(
                local_state, buffer, source_local_first_index, source_local_last_index,
                datatype, static_cast<yampi::rank>(target_global_index),
                communicator, environment);
            }

            using ::ket::mpi::permutate;
            permutate(permutation, qubits[0u], local_swap_qubit0);
            permutate(permutation, qubits[1u], local_swap_qubit1);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }
        };

        template <>
        struct maybe_interchange_qubits<
          3u, ::ket::mpi::utility::policy::general_mpi>
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, 3u> const& qubits,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits> const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));


            typedef ket::qubit<StateInteger, BitInteger> qubit_type;

            BitInteger const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));
            qubit_type const permutated_global_swap_qubit0
              = permutation[qubits[0u]];
            qubit_type const permutated_global_swap_qubit1
              = permutation[qubits[1u]];
            qubit_type const permutated_global_swap_qubit2
              = permutation[qubits[2u]];


            if (static_cast<BitInteger>(permutated_global_swap_qubit0)
                < num_local_qubits)
            {
              KET_array<qubit_type, 2u> new_qubits = { qubits[1u], qubits[2u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[0u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  2u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_double_qubit;

              maybe_interchange_double_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, communicator, environment);

              return;
            }
            else if (static_cast<BitInteger>(permutated_global_swap_qubit1)
                  < num_local_qubits)
            {
              KET_array<qubit_type, 2u> new_qubits = { qubits[0u], qubits[2u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[1u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  2u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_double_qubit;

              maybe_interchange_double_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, communicator, environment);

              return;
            }
            else if (static_cast<BitInteger>(permutated_global_swap_qubit2)
                  < num_local_qubits)
            {
              KET_array<qubit_type, 2u> new_qubits = { qubits[0u], qubits[1u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[2u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  2u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_double_qubit;

              maybe_interchange_double_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, communicator, environment);

              return;
            }


            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<3>", environment);

# ifndef NDEBUG
            boost::optional<yampi::rank> const maybe_io_rank = yampi::lowest_io_process(environment);
            yampi::rank const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            //  Swaps between xxbxb'xb''xx|cc'c''xxxxxxxx and
            // xxcxc'xx|bb'xxxxxxxx (c = b or ~b). Upper qubits are global
            // qubits representing MPI rank. Lower qubits are local qubits
            // representing memory address. The first two upper qubits in the
            // local qubits are "local swap qubits". Two consecutive bits in
            // global qubits and the "local swap qubits" would be swapped.

            qubit_type const permutated_local_swap_qubit0(num_local_qubits-1u);
            qubit_type const local_swap_qubit0
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit0);
            qubit_type const permutated_local_swap_qubit1(num_local_qubits-2u);
            qubit_type const local_swap_qubit1
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit1);
            qubit_type const permutated_local_swap_qubit2(num_local_qubits-3u);
            qubit_type const local_swap_qubit2
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit2);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG

            // xxbxb'xb''xx(|xxxxxxxxxx)
            StateInteger const source_global_index
              = static_cast<StateInteger>(communicator.rank(environment).mpi_rank());

            for (StateInteger target_global_mask = 1u;
                 target_global_mask < ::ket::utility::integer_exp2<StateInteger>(3u);
                 ++target_global_mask)
            {
              StateInteger const target_global_mask0
                = target_global_mask bitand static_cast<StateInteger>(1u);
              StateInteger const target_global_mask1
                = (target_global_mask bitand static_cast<StateInteger>(2u)) >> 1;
              StateInteger const target_global_mask2
                = (target_global_mask bitand static_cast<StateInteger>(4u)) >> 2;

              // xxcxc'xc''xx(|xxxxxxxxxx) (c = b or ~b, except for (c, c', c'') = (b, b', b''))
              StateInteger const target_global_index
                = source_global_index
                  xor
                  (((target_global_mask0 << permutated_global_swap_qubit0)
                    >> num_local_qubits)
                   bitor
                   ((target_global_mask1 << permutated_global_swap_qubit1)
                    >> num_local_qubits)
                   bitor
                   ((target_global_mask2 << permutated_global_swap_qubit2)
                    >> num_local_qubits));

              // (0000000|)c0000000000
              StateInteger const source_local_first_index0
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit0))
                  >> (permutated_global_swap_qubit0-permutated_local_swap_qubit0);
              // (0000000|)0c'000000000
              StateInteger const source_local_first_index1
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit1))
                  >> (permutated_global_swap_qubit1-permutated_local_swap_qubit1);
              // (0000000|)00c''00000000
              StateInteger const source_local_first_index2
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit2))
                  >> (permutated_global_swap_qubit2-permutated_local_swap_qubit2);
              // (0000000|)cc'c''00000000
              StateInteger const source_local_first_index
                = source_local_first_index0 bitor source_local_first_index1
                  bitor source_local_first_index2;
              // (0000000|)0001111111
              StateInteger const prev_last_mask
                = (static_cast<StateInteger>(1u) << permutated_local_swap_qubit2)
                  - static_cast<StateInteger>(1u);
              // (0000000|)cc'c''11111111 + 1
              StateInteger const source_local_last_index
                = (source_local_first_index bitor prev_last_mask)
                  + static_cast<StateInteger>(1u);

              ::ket::mpi::utility::log_with_time_guard<char> print(
                "interchange_qubits<3>::swap", environment);

              ::ket::mpi::utility::detail::interchange_qubits(
                local_state, buffer, source_local_first_index, source_local_last_index,
                static_cast<yampi::rank>(target_global_index), communicator, environment);
            }

            using ::ket::mpi::permutate;
            permutate(permutation, qubits[0u], local_swap_qubit0);
            permutate(permutation, qubits[1u], local_swap_qubit1);
            permutate(permutation, qubits[2u], local_swap_qubit2);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }

          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger,
            std::size_t num_unswappable_qubits, typename Allocator, typename BufferAllocator, typename DerivedDatatype>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, 3u> const& qubits,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>, num_unswappable_qubits> const&
              unswappable_qubits,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
            std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
            yampi::datatype_base<DerivedDatatype> const& datatype,
            yampi::communicator const& communicator,
            yampi::environment const& environment)
          {
            static_assert(
              KET_is_unsigned<StateInteger>::value, "StateInteger should be unsigned");
            static_assert(
              KET_is_unsigned<BitInteger>::value, "BitInteger should be unsigned");

            assert(communicator.size(environment) > 1);
            assert(
              ::ket::utility::integer_exp2<StateInteger>(
                ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)))
              == boost::size(local_state));


            typedef ket::qubit<StateInteger, BitInteger> qubit_type;

            BitInteger const num_local_qubits
              = ::ket::utility::integer_log2<BitInteger>(boost::size(local_state));
            qubit_type const permutated_global_swap_qubit0
              = permutation[qubits[0u]];
            qubit_type const permutated_global_swap_qubit1
              = permutation[qubits[1u]];
            qubit_type const permutated_global_swap_qubit2
              = permutation[qubits[2u]];


            if (static_cast<BitInteger>(permutated_global_swap_qubit0)
                < num_local_qubits)
            {
              KET_array<qubit_type, 2u> new_qubits = { qubits[1u], qubits[2u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[0u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  2u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_double_qubit;

              maybe_interchange_double_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);

              return;
            }
            else if (static_cast<BitInteger>(permutated_global_swap_qubit1)
                  < num_local_qubits)
            {
              KET_array<qubit_type, 2u> new_qubits = { qubits[0u], qubits[2u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[1u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  2u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_double_qubit;

              maybe_interchange_double_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);

              return;
            }
            else if (static_cast<BitInteger>(permutated_global_swap_qubit2)
                  < num_local_qubits)
            {
              KET_array<qubit_type, 2u> new_qubits = { qubits[0u], qubits[1u] };
              KET_array<qubit_type, num_unswappable_qubits+1u> new_unswappable_qubits;
              std::copy(
                ::ket::utility::begin(unswappable_qubits),
                ::ket::utility::end(unswappable_qubits),
                ::ket::utility::begin(new_unswappable_qubits));
              new_unswappable_qubits.back() = qubits[2u];

              typedef
                ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
                  2u, ::ket::mpi::utility::policy::general_mpi>
                maybe_interchange_double_qubit;

              maybe_interchange_double_qubit::call(
                parallel_policy, local_state, new_qubits, new_unswappable_qubits,
                permutation, buffer, datatype, communicator, environment);

              return;
            }


            ::ket::mpi::utility::log_with_time_guard<char> print(
              "interchange_qubits<3>", environment);

# ifndef NDEBUG
            boost::optional<yampi::rank> const maybe_io_rank = yampi::lowest_io_process(environment);
            yampi::rank const my_rank = yampi::communicator(yampi::world_communicator_t()).rank(environment);
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation before changing qubits] " << permutation << std::endl;
# endif // NDEBUG

            //  Swaps between xxbxb'xb''xx|cc'c''xxxxxxxx and
            // xxcxc'xx|bb'xxxxxxxx (c = b or ~b). Upper qubits are global
            // qubits representing MPI rank. Lower qubits are local qubits
            // representing memory address. The first two upper qubits in the
            // local qubits are "local swap qubits". Two consecutive bits in
            // global qubits and the "local swap qubits" would be swapped.

            qubit_type const permutated_local_swap_qubit0(num_local_qubits-1u);
            qubit_type const local_swap_qubit0
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit0);
            qubit_type const permutated_local_swap_qubit1(num_local_qubits-2u);
            qubit_type const local_swap_qubit1
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit1);
            qubit_type const permutated_local_swap_qubit2(num_local_qubits-3u);
            qubit_type const local_swap_qubit2
              = ::ket::mpi::utility::general_mpi_detail::make_local_swap_qubit_swappable(
                  parallel_policy, local_state, permutation,
                  unswappable_qubits, permutated_local_swap_qubit2);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local swap qubits] " << permutation << std::endl;
# endif // NDEBUG

            // xxbxb'xb''xx(|xxxxxxxxxx)
            StateInteger const source_global_index
              = static_cast<StateInteger>(communicator.rank(environment).mpi_rank());

            for (StateInteger target_global_mask = 1u;
                 target_global_mask < ::ket::utility::integer_exp2<StateInteger>(3u);
                 ++target_global_mask)
            {
              StateInteger const target_global_mask0
                = target_global_mask bitand static_cast<StateInteger>(1u);
              StateInteger const target_global_mask1
                = (target_global_mask bitand static_cast<StateInteger>(2u)) >> 1;
              StateInteger const target_global_mask2
                = (target_global_mask bitand static_cast<StateInteger>(4u)) >> 2;

              // xxcxc'xc''xx(|xxxxxxxxxx) (c = b or ~b, except for (c, c', c'') = (b, b', b''))
              StateInteger const target_global_index
                = source_global_index
                  xor
                  (((target_global_mask0 << permutated_global_swap_qubit0)
                    >> num_local_qubits)
                   bitor
                   ((target_global_mask1 << permutated_global_swap_qubit1)
                    >> num_local_qubits)
                   bitor
                   ((target_global_mask2 << permutated_global_swap_qubit2)
                    >> num_local_qubits));

              // (0000000|)c0000000000
              StateInteger const source_local_first_index0
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit0))
                  >> (permutated_global_swap_qubit0-permutated_local_swap_qubit0);
              // (0000000|)0c'000000000
              StateInteger const source_local_first_index1
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit1))
                  >> (permutated_global_swap_qubit1-permutated_local_swap_qubit1);
              // (0000000|)00c''00000000
              StateInteger const source_local_first_index2
                = ((target_global_index << num_local_qubits)
                   bitand ::ket::utility::integer_exp2<StateInteger>(
                            permutated_global_swap_qubit2))
                  >> (permutated_global_swap_qubit2-permutated_local_swap_qubit2);
              // (0000000|)cc'c''00000000
              StateInteger const source_local_first_index
                = source_local_first_index0 bitor source_local_first_index1
                  bitor source_local_first_index2;
              // (0000000|)0001111111
              StateInteger const prev_last_mask
                = (static_cast<StateInteger>(1u) << permutated_local_swap_qubit2)
                  - static_cast<StateInteger>(1u);
              // (0000000|)cc'c''11111111 + 1
              StateInteger const source_local_last_index
                = (source_local_first_index bitor prev_last_mask)
                  + static_cast<StateInteger>(1u);

              ::ket::mpi::utility::log_with_time_guard<char> print(
                "interchange_qubits<3>::swap", environment);

              ::ket::mpi::utility::detail::interchange_qubits(
                local_state, buffer, source_local_first_index, source_local_last_index,
                datatype, static_cast<yampi::rank>(target_global_index),
                communicator, environment);
            }

            using ::ket::mpi::permutate;
            permutate(permutation, qubits[0u], local_swap_qubit0);
            permutate(permutation, qubits[1u], local_swap_qubit1);
            permutate(permutation, qubits[2u], local_swap_qubit2);

# ifndef NDEBUG
            if (maybe_io_rank && my_rank == *maybe_io_rank)
              std::clog << "[permutation after changing local/global qubits] " << permutation << std::endl;
# endif // NDEBUG
          }
        };


        template <typename MpiPolicy, typename LocalState_>
        struct for_each_local_range
        {
          template <typename LocalState, typename Function>
          static LocalState& call(
            LocalState& local_state, KET_RVALUE_REFERENCE_OR_COPY(Function) function);

          template <typename LocalState, typename Function>
          static LocalState const& call(
            LocalState const& local_state, KET_RVALUE_REFERENCE_OR_COPY(Function) function);
        };

        template <typename LocalState_>
        struct for_each_local_range< ::ket::mpi::utility::policy::general_mpi, LocalState_>
        {
          template <typename LocalState, typename Function>
          static LocalState& call(
            LocalState& local_state, KET_RVALUE_REFERENCE_OR_COPY(Function) function)
          {
            function(::ket::utility::begin(local_state), ::ket::utility::end(local_state));
            return local_state;
          }

          template <typename LocalState, typename Function>
          static LocalState& call(
            LocalState const& local_state, KET_RVALUE_REFERENCE_OR_COPY(Function) function)
          {
            function(::ket::utility::begin(local_state), ::ket::utility::end(local_state));
            return local_state;
          }
        };


        template <typename MpiPolicy>
        struct rank_index_to_qubit_value
        {
          template <typename LocalState, typename StateInteger>
          static StateInteger call(
            LocalState const& local_state,
            yampi::rank const rank, StateInteger const index);
        };

        template <>
        struct rank_index_to_qubit_value< ::ket::mpi::utility::policy::general_mpi>
        {
          template <typename LocalState, typename StateInteger>
          static StateInteger call(
            LocalState const& local_state,
            yampi::rank const rank, StateInteger const index)
          { return rank.mpi_rank() * boost::size(local_state) + index; }
        };


        template <typename MpiPolicy>
        struct qubit_value_to_rank_index
        {
          template <typename LocalState, typename StateInteger>
          static std::pair<yampi::rank, StateInteger> call(
            LocalState const& local_state, StateInteger const qubit_value);
        };

        template <>
        struct qubit_value_to_rank_index< ::ket::mpi::utility::policy::general_mpi>
        {
          template <typename LocalState, typename StateInteger>
          static std::pair<yampi::rank, StateInteger> call(
            LocalState const& local_state, StateInteger const qubit_value)
          {
            return std::make_pair(
              static_cast<yampi::rank>(qubit_value / boost::size(local_state)),
              qubit_value % boost::size(local_state));
          }
        };


# ifdef KET_USE_DIAGONAL_LOOP
        // 170607-: 
#   ifdef BOOST_NO_CXX11_LAMBDAS
        namespace diagonal_loop_detail
        {
          template <typename Result, typename StateInteger>
          class call_function0_or_function1
          {
            boost::function<Result(StateInteger)> function0_;
            boost::function<Result(StateInteger)> function1_;
            StateInteger mask_;

            BOOST_STATIC_CONSTEXPR StateInteger zero_state_integer_ = 0u;

           public:
            typedef Result result_type;

            template <typename Function0, typename Function1>
            call_function0_or_function1(
              KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,
              KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,
              StateInteger const mask)
              : function0_(KET_FORWARD_OR_COPY(Function0, function0)),
                function1_(KET_FORWARD_OR_COPY(Function1, function1)),
                mask_{mask}
            { }

            Result operator()(StateInteger const state_integer, int const)
            {
              if ((state_integer bitand mask_) == zero_state_integer_)
                return function0_(state_integer);
              else
                return function1_(state_integer);
            }

            Result operator()(StateInteger const state_integer, int const) const
            {
              if ((state_integer bitand mask_) == zero_state_integer_)
                return function0_(state_integer);
              else
                return function1_(state_integer);
            }
          };

          template <typename Result, typename Function0, typename Function1, typename StateInteger>
          inline
          ::ket::mpi::utility::dispatch::diagonal_loop_detail::call_function0_or_function1<Result, StateInteger>
          make_call_function0_or_function1(
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,
            StateInteger const mask)
          {
            return ::ket::mpi::utility::dispatch::diagonal_loop_detail::call_function0_or_function1<Result, StateInteger>(
              KET_FORWARD_OR_COPY(Function0, function0),
              KET_FORWARD_OR_COPY(function1, function1), mask);
          }


          template <typename StateInteger, typename BitInteger>
          struct generate_control_qubit_mask
          {
            BOOST_STATIC_CONSTEXPR StateInteger one_state_integer_ = 1u;

            typedef StateInteger result_type;

            StateInteger operator()(
              StateInteger const& partial_mask,
              ::ket::qubit<StateInteger, BitInteger> const& control_qubit) const
            { return partial_mask bitor (one_state_integer << control_qubit); }
          };


          template <
            typename Result,
            typename StateInteger, typename BitInteger, std::size_t num_local_control_qubits>
          class for_each_inside
          {
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
            typedef KET_array<qubit_type, num_local_control_qubits> qubits_type;

            std::function<Result(StateInteger)> function_;
            qubits_type sorted_local_permutated_control_qubits_;
            StateInteger mask_;

            BOOST_STATIC_CONSTEXPR StateInteger one_state_integer_ = 1u;

           public:
            typedef Result result_type;

            template <typename Function>
            for_each_inside(
              KET_RVALUE_REFERENCE_OR_COPY(Function) function,
              qubits_type const& sorted_local_permutated_control_qubits,
              StateInteger const mask)
              : function_(function),
                sorted_local_permutated_control_qubits_(sorted_local_permutated_control_qubits),
                mask_(mask)
            { }

            Result operator()(StateInteger state_integer, int const)
            {
              // xxx0x0xxx0xx
#     ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
              for (qubit_type const& qubit: sorted_local_permutated_control_qubits_)
              {
                StateInteger const lower_mask = (one_state_integer_ << qubit) - one_state_integer_;
                StateInteger const upper_mask = compl lower_mask;
                state_integer
                  = (state_integer bitand lower_mask)
                    bitor ((state_integer bitand upper_mask) << 1u);
              }
#     else // BOOST_NO_CXX11_RANGE_BASED_FOR
              typedef typename ::ket::utility::meta::const_iterator_of<qubits_type>::type iterator;

              iterator const last = ::ket::utility::end(sorted_local_permutated_control_qubits_);
              for (iterator iter = ::ket::utility::begin(sorted_local_permutated_control_qubits_);
                   iter != last; ++iter)
              {
                StateInteger const lower_mask = (one_state_integer_ << *iter) - one_state_integer_;
                StateInteger const upper_mask = compl lower_mask;
                state_integer
                  = (state_integer bitand lower_mask)
                    bitor ((state_integer bitand upper_mask) << 1u);
              }
#     endif // BOOST_NO_CXX11_RANGE_BASED_FOR

              // function(xxx1x1xxx1xx)
              function(state_integer bitor mask);
            }

            Result operator()(StateInteger state_integer, int const) const
            {
              // xxx0x0xxx0xx
#     ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
              for (qubit_type const& qubit: sorted_local_permutated_control_qubits_)
              {
                StateInteger const lower_mask = (one_state_integer_ << qubit) - one_state_integer_;
                StateInteger const upper_mask = compl lower_mask;
                state_integer
                  = (state_integer bitand lower_mask)
                    bitor ((state_integer bitand upper_mask) << 1u);
              }
#     else // BOOST_NO_CXX11_RANGE_BASED_FOR
              typedef typename ::ket::utility::meta::const_iterator_of<qubits_type>::type iterator;

              iterator const last = ::ket::utility::end(sorted_local_permutated_control_qubits_);
              for (iterator iter = ::ket::utility::begin(sorted_local_permutated_control_qubits_);
                   iter != last; ++iter)
              {
                StateInteger const lower_mask = (one_state_integer_ << *iter) - one_state_integer_;
                StateInteger const upper_mask = compl lower_mask;
                state_integer
                  = (state_integer bitand lower_mask)
                    bitor ((state_integer bitand upper_mask) << 1u);
              }
#     endif // BOOST_NO_CXX11_RANGE_BASED_FOR

              // function(xxx1x1xxx1xx)
              function(state_integer bitor mask);
            }
          };

          template <
            typename Result, typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          inline
          ::ket::mpi::utility::dispatch::diagonal_loop_detail::for_each_inside<
            Result, StateInteger, BitInteger, num_local_control_qubits>
          make_for_each_inside(
            KET_RVALUE_REFERENCE_OR_COPY(Function) function,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& sorted_local_permutated_control_qubits,
            StateInteger const mask)
          {
            return ::ket::mpi::utility::dispatch::diagonal_loop_detail::for_each_inside<
              Result, StateInteger, BitInteger, num_local_control_qubits>(
              KET_FORWARD_OR_COPY(Function, function),
              sorted_local_permutated_control_qubits, mask);
          }
        } // namespace diagonal_loop_detail
#   endif

        template <typename MpiPolicy>
        struct diagonal_loop
        {
#   ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            ParallelPolicy const, LocalState const& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,
            ControlQubits... control_qubits);
#   else // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#     define KET_DIAGONAL_LOOP_CALL(z, n, _) \
          template <\
            typename ParallelPolicy, typename LocalState,\
            typename StateInteger, typename BitInteger, typename Allocator,\
            typename Function0, typename Function1 BOOST_PP_ENUM_TRAILING_PARAMS(n, typename ControlQubit)>\
          static void call(\
            ParallelPolicy const, LocalState const& local_state,\
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,\
            yampi::communicator const& communicator,\
            yampi::environment const& environment,\
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,\
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,\
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1\
            BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, ControlQubit, const control_qubit));

          BOOST_PP_REPEAT(KET_MAX_NUM_CONTROL_QUBITS, KET_DIAGONAL_LOOP_CALL, _)

#     undef KET_DIAGONAL_LOOP_CALL
#   endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
        };

        template <>
        struct diagonal_loop< ::ket::mpi::utility::policy::general_mpi >
        {
#   ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
          template <
            typename ParallelPolicy, typename LocalState,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1, typename... ControlQubits>
          static void call(
            ParallelPolicy const parallel_policy, LocalState const& local_state,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::communicator const& communicator,
            yampi::environment const& environment,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,
            ControlQubits... control_qubits)
          {
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

            KET_array<qubit_type, 0u> local_permutated_control_qubits;

            qubit_type const least_global_permutated_qubit(
              ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)));

            call_impl(
              parallel_policy, permutation, communicator.rank(environment),
              least_global_permutated_qubit, target_qubit,
              KET_FORWARD_OR_COPY(Function0, function0),
              KET_FORWARD_OR_COPY(Function1, function1),
              local_permutated_control_qubits, control_qubits...);
          }
#   else // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#     define KET_DIAGONAL_LOOP_CALL(z, n, _) \
          template <\
            typename ParallelPolicy, typename LocalState,\
            typename StateInteger, typename BitInteger, typename Allocator,\
            typename Function0, typename Function1 BOOST_PP_ENUM_TRAILING_PARAMS(n, typename ControlQubit)>\
          static void call(\
            ParallelPolicy const parallel_policy, LocalState const& local_state,\
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,\
            yampi::communicator const& communicator,\
            yampi::environment const& environment,\
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,\
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,\
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1\
            BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, ControlQubit, const control_qubit))\
          {\
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;\
\
            KET_array<qubit_type, 0u> local_permutated_control_qubits;\
\
            qubit_type const least_global_permutated_qubit(\
              ::ket::utility::integer_log2<BitInteger>(boost::size(local_state)));\
\
            call_impl(\
              parallel_policy, permutation, communicator.rank(environment),\
              least_global_permutated_qubit, target_qubit,\
              KET_FORWARD_OR_COPY(Function0, function0),\
              KET_FORWARD_OR_COPY(Function1, function1),\
              local_permutated_control_qubits\
              BOOST_PP_ENUM_TRAILING_PARAMS(n, control_qubit));\
          }

          BOOST_PP_REPEAT(KET_MAX_NUM_CONTROL_QUBITS, KET_DIAGONAL_LOOP_CALL, _)

#     undef KET_DIAGONAL_LOOP_CALL
#   endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

         private:
#   ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_local_control_qubits, typename... ControlQubits>
          static void call_impl(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const rank,
            ::ket::qubit<StateInteger, BitInteger> const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& local_permutated_control_qubits,
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit,
            ControlQubits... control_qubits)
          {
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

            qubit_type const permutated_control_qubit = permutation[control_qubit.qubit()];

            if (permutated_control_qubit < least_global_permutated_qubit)
            {
              KET_array<qubit_type, num_local_control_qubits+1u> new_local_permutated_control_qubits;
              std::copy(
                ::ket::utility::begin(local_permutated_control_qubits),
                ::ket::utility::end(local_permutated_control_qubits),
                ::ket::utility::begin(new_local_permutated_control_qubits));
              new_local_permutated_control_qubits.back() = permutated_control_qubit;

              call_impl(
                parallel_policy, permutation, rank,
                least_global_permutated_qubit, target_qubit,
                KET_FORWARD_OR_COPY(Function0, function0),
                KET_FORWARD_OR_COPY(Function1, function1),
                new_local_permutated_control_qubits, control_qubits...);
            }
            else
            {
              BOOST_CONSTEXPR_OR_CONST StateInteger zero_state_integer = 0u;
              BOOST_CONSTEXPR_OR_CONST StateInteger one_state_integer = 1u;

              StateInteger const mask
                = one_state_integer << (permutated_control_qubit - least_global_permutated_qubit);

              if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) != zero_state_integer)
                call_impl(
                  parallel_policy, permutation, rank,
                  least_global_permutated_qubit, target_qubit,
                  KET_FORWARD_OR_COPY(Function0, function0),
                  KET_FORWARD_OR_COPY(Function1, function1),
                  local_permutated_control_qubits, control_qubits...);
            }
          }
#   else // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#     define KET_DIAGONAL_LOOP_CALL(z, n, _) \
          template <\
            typename ParallelPolicy,\
            typename StateInteger, typename BitInteger, typename Allocator,\
            typename Function0, typename Function1,\
            std::size_t num_local_control_qubits typename Function1 BOOST_PP_ENUM_TRAILING_PARAMS(n, typename ControlQubit)>\
          static void call_impl(\
            ParallelPolicy const parallel_policy,\
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,\
            yampi::rank const rank,\
            ::ket::qubit<StateInteger, BitInteger> const least_global_permutated_qubit,\
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,\
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,\
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,\
            KET_array<\
              ::ket::qubit<StateInteger, BitInteger>,\
              num_local_control_qubits> const& local_permutated_control_qubits,\
            ::ket::control< ::ket::qubit<StateInteger, BitInteger> > const control_qubit\
            BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, ControlQubit, const control_qubit))\
          {\
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;\
\
            qubit_type const permutated_control_qubit = permutation[control_qubit.qubit()];\
\
            if (permutated_control_qubit < least_global_qubit)\
            {\
              KET_array<qubit_type, num_local_control_qubits+1u> new_local_permutated_control_qubits;\
              std::copy(\
                ::ket::utility::begin(local_permutated_control_qubits),\
                ::ket::utility::end(local_permutated_control_qubits),\
                ::ket::utility::begin(new_local_permutated_control_qubits));\
              new_local_permutated_control_qubits.back() = permutated_control_qubit;\
\
              call_impl(\
                parallel_policy, permutation, rank,\
                least_global_permutated_qubit, target_qubit,\
                KET_FORWARD_OR_COPY(Function0, function0),\
                KET_FORWARD_OR_COPY(Function1, function1),\
                new_local_permutated_control_qubits\
                BOOST_PP_ENUM_TRAILING_PARAMS(n, control_qubit));\
            }\
            else\
            {\
              BOOST_CONSTEXPR_OR_CONST StateInteger zero_state_integer = 0u;\
              BOOST_CONSTEXPR_OR_CONST StateInteger one_state_integer = 1u;\
\
              StateInteger const mask\
                = one_state_integer << (permutated_control_qubit - least_global_permutated_qubit);\
\
              if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) != zero_state_integer)\
                call_impl(\
                  parallel_policy, permutation, rank,\
                  least_global_permutated_qubit, target_qubit,\
                  KET_FORWARD_OR_COPY(Function0, function0),\
                  KET_FORWARD_OR_COPY(Function1, function1),\
                  local_permutated_control_qubits\
                  BOOST_PP_ENUM_TRAILING_PARAMS(n, control_qubit));\
            }\
          }

          BOOST_PP_REPEAT(BOOST_PP_DEC(KET_MAX_NUM_CONTROL_QUBITS), KET_DIAGONAL_LOOP_CALL, _)

#     undef KET_DIAGONAL_LOOP_CALL
#   endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES

          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger, typename Allocator,
            typename Function0, typename Function1,
            std::size_t num_local_control_qubits>
          static void call_impl(
            ParallelPolicy const parallel_policy,
            ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
            yampi::rank const rank,
            ::ket::qubit<StateInteger, BitInteger> const least_global_permutated_qubit,
            ::ket::qubit<StateInteger, BitInteger> const target_qubit,
            KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,
            KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& local_permutated_control_qubits)
          {
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;
            qubit_type const permutated_target_qubit = permutation[target_qubit];

              BOOST_CONSTEXPR_OR_CONST StateInteger one_state_integer = 1u;

            StateInteger const last_integer
              = (one_state_integer << least_global_permutated_qubit)
                >> boost::size(local_permutated_control_qubits);

            if (permutated_target_qubit < least_global_permutated_qubit)
            {
              StateInteger const mask = one_state_integer << permutated_target_qubit;

#   ifndef BOOST_NO_CXX11_LAMBDAS
              for_each(
                parallel_policy, last_integer, local_permutated_control_qubits,
                [&function0, &function1, mask](StateInteger const state_integer)
                {
                  BOOST_CONSTEXPR_OR_CONST StateInteger zero_state_integer = 0u;

                  if ((state_integer bitand mask) == zero_state_integer)
                    function0(state_integer);
                  else
                    function1(state_integer);
                });
#   else // BOOST_NO_CXX11_LAMBDAS
              for_each(
                parallel_policy, last_integer, local_permutated_control_qubits,
                ::ket::mpi::utility::dispatch::diagonal_loop_detail::make_call_function0_or_function1<void>(
                  KET_FORWARD_OR_COPY(Function0, function0),
                  KET_FORWARD_OR_COPY(Function1, function1), mask));
#   endif // BOOST_NO_CXX11_LAMBDAS
            }
            else
            {
              StateInteger const mask
                = one_state_integer << (permutated_target_qubit - least_global_permutated_qubit);

              BOOST_CONSTEXPR_OR_CONST StateInteger zero_state_integer = 0u;

              if ((static_cast<StateInteger>(rank.mpi_rank()) bitand mask) == zero_state_integer)
                for_each(
                  parallel_policy, last_integer, local_permutated_control_qubits,
                  KET_FORWARD_OR_COPY(Function0, function0));
              else
                for_each(
                  parallel_policy, last_integer, local_permutated_control_qubits,
                  KET_FORWARD_OR_COPY(Function1, function1));
            }
          }

          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          static void for_each(
            ParallelPolicy const parallel_policy, StateInteger const last_integer,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& local_permutated_control_qubits,
            KET_RVALUE_REFERENCE_OR_COPY(Function) function)
          {
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

            KET_array<qubit_type, num_local_control_qubits> sorted_local_permutated_control_qubits
              = local_permutated_control_qubits;
            std::sort(
              ::ket::utility::begin(sorted_local_permutated_control_qubits),
              ::ket::utility::end(sorted_local_permutated_control_qubits));

            for_each_impl(
              parallel_policy, last_integer, sorted_local_permutated_control_qubits,
              KET_FORWARD_OR_COPY(Function, function));
          }

          template <
            typename ParallelPolicy,
            typename StateInteger, typename BitInteger,
            std::size_t num_local_control_qubits, typename Function>
          static void for_each_impl(
            ParallelPolicy const parallel_policy, StateInteger const last_integer,
            KET_array<
              ::ket::qubit<StateInteger, BitInteger>,
              num_local_control_qubits> const& sorted_local_permutated_control_qubits,
            KET_RVALUE_REFERENCE_OR_COPY(Function) function)
          {
            typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

            BOOST_CONSTEXPR_OR_CONST StateInteger zero_state_integer = 0u;

            // 000101000100
#   ifndef BOOST_NO_CXX11_LAMBDAS
            StateInteger const mask
              = boost::accumulate(
                  sorted_local_permutated_control_qubits, zero_state_integer,
                  [](StateInteger const& partial_mask, qubit_type const& control_qubit)
                  {
                    BOOST_CONSTEXPR_OR_CONST StateInteger one_state_integer = 1u;

                    return partial_mask bitor (one_state_integer << control_qubit);
                  });
#   else // BOOST_NO_CXX11_LAMBDAS
            StateInteger const mask
              = std::accumulate(
                  ::ket::utility::begin(sorted_local_permutated_control_qubits),
                  ::ket::utility::end(sorted_local_permutated_control_qubits),
                  zero_state_integer,
                  ::ket::mpi::utility::dispatch::diagonal_loop_detail::generate_control_qubit_mask<StateInteger, BitInteger>());
#   endif // BOOST_NO_CXX11_LAMBDAS

            using ::ket::utility::loop_n;
#   ifndef BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy, last_integer,
              [&function, &sorted_local_permutated_control_qubits, mask](StateInteger state_integer, int const)
              {
                BOOST_CONSTEXPR_OR_CONST StateInteger one_state_integer = 1u;

                // xxx0x0xxx0xx
#     ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
                for (qubit_type const& qubit: sorted_local_permutated_control_qubits)
                {
                  StateInteger const lower_mask = (one_state_integer << qubit) - one_state_integer;
                  StateInteger const upper_mask = compl lower_mask;
                  state_integer
                    = (state_integer bitand lower_mask)
                      bitor ((state_integer bitand upper_mask) << 1u);
                }
#     else // BOOST_NO_CXX11_RANGE_BASED_FOR
                typedef
                  typename ::ket::utility::meta::const_iterator_of<
                    KET_array<
                      ::ket::qubit<StateInteger, BitInteger>,
                      num_local_control_qubits>
                  >::type
                  iterator;

                iterator const last = ::ket::utility::end(sorted_local_permutated_control_qubits);
                for (iterator iter = ::ket::utility::begin(sorted_local_permutated_control_qubits);
                     iter != last; ++iter)
                {
                  StateInteger const lower_mask = (one_state_integer << *iter) - one_state_integer;
                  StateInteger const upper_mask = compl lower_mask;
                  state_integer
                    = (state_integer bitand lower_mask)
                      bitor ((state_integer bitand upper_mask) << 1u);
                }
#     endif // BOOST_NO_CXX11_RANGE_BASED_FOR

                // function(xxx1x1xxx1xx)
                function(state_integer bitor mask);
              });
#   else // BOOST_NO_CXX11_LAMBDAS
            loop_n(
              parallel_policy, last_integer,
              ::ket::mpi::utility::dispatch::diagonal_loop_detail::make_for_each_inside(
                KET_FORWARD_OR_COPY(Function, function),
                sorted_local_permutated_control_qubits, mask));
#   endif // BOOST_NO_CXX11_LAMBDAS
          }
        };
# endif // KET_USE_DIAGONAL_LOOP
      } // namespace dispatch


      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        MpiPolicy const, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        KET_array<
          ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        typedef
          ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
            num_qubits_of_operation, MpiPolicy>
          maybe_interchange_qubits_impl;
        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

        KET_array<qubit_type, 0u> unswappable_qubits;

        maybe_interchange_qubits_impl::call(
          parallel_policy,
          local_state, qubits, unswappable_qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        MpiPolicy const, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        KET_array<
          ::ket::qubit<StateInteger, BitInteger>, num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        typedef
          ::ket::mpi::utility::dispatch::maybe_interchange_qubits<
            num_qubits_of_operation, MpiPolicy>
          maybe_interchange_qubits_impl;
        typedef ::ket::qubit<StateInteger, BitInteger> qubit_type;

        KET_array<qubit_type, 0u> unswappable_qubits;

        maybe_interchange_qubits_impl::call(
          parallel_policy,
          local_state, qubits, unswappable_qubits, permutation,
          buffer, datatype, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        KET_array<
          ::ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename ParallelPolicy, typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        ParallelPolicy const parallel_policy,
        LocalState& local_state,
        KET_array<
          ::ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(), parallel_policy,
          local_state, qubits, permutation, buffer, datatype, communicator, environment);
      }

      template <
        typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator>
      void maybe_interchange_qubits(
        LocalState& local_state,
        KET_array<
          ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubits, permutation, buffer, communicator, environment);
      }

      template <
        typename LocalState,
        typename StateInteger, typename BitInteger, std::size_t num_qubits_of_operation,
        typename Allocator, typename BufferAllocator, typename DerivedDatatype>
      void maybe_interchange_qubits(
        LocalState& local_state,
        KET_array<
          ket::qubit<StateInteger, BitInteger>,
          num_qubits_of_operation> const& qubits,
        ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector<typename boost::range_value<LocalState>::type, BufferAllocator>& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator,
        yampi::environment const& environment)
      {
        maybe_interchange_qubits(
          ::ket::mpi::utility::policy::make_general_mpi(),
          ::ket::utility::policy::make_sequential(),
          local_state, qubits, permutation, buffer, datatype, communicator, environment);
      }


      template <typename MpiPolicy, typename LocalState, typename Function>
      inline LocalState& for_each_local_range(
        MpiPolicy const, LocalState& local_state,
        KET_RVALUE_REFERENCE_OR_COPY(Function) function)
      {
        return ::ket::mpi::utility::dispatch::for_each_local_range<MpiPolicy, LocalState>::call(
          local_state, KET_FORWARD_OR_COPY(Function, function));
      }

      template <typename MpiPolicy, typename LocalState, typename Function>
      inline LocalState const& for_each_local_range(
        MpiPolicy const, LocalState const& local_state,
        KET_RVALUE_REFERENCE_OR_COPY(Function) function)
      {
        return ::ket::mpi::utility::dispatch::for_each_local_range<MpiPolicy, LocalState>::call(
          local_state, KET_FORWARD_OR_COPY(Function, function));
      }


      template <
        typename MpiPolicy, typename LocalState, typename StateInteger>
      inline StateInteger rank_index_to_qubit_value(
        MpiPolicy const, LocalState const& local_state, yampi::rank const rank,
        StateInteger const index)
      {
        return ::ket::mpi::utility::dispatch::rank_index_to_qubit_value<MpiPolicy>::call(
          local_state, rank, index);
      }


      template <
        typename MpiPolicy, typename LocalState, typename StateInteger>
      inline std::pair<yampi::rank, StateInteger> qubit_value_to_rank_index(
        MpiPolicy const, LocalState const& local_state, StateInteger const qubit_value)
      {
        return ::ket::mpi::utility::dispatch::qubit_value_to_rank_index<MpiPolicy>::call(
          local_state, qubit_value);
      }


# ifdef KET_USE_DIAGONAL_LOOP
      // 170607-: 
#   ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState,
        typename StateInteger, typename BitInteger, typename Allocator,
        typename Function0, typename Function1, typename... ControlQubits>
      inline void diagonal_loop(
        MpiPolicy const, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,
        yampi::communicator const& communicator,
        yampi::environment const& environment,
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,
        KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,
        KET_RVALUE_REFERENCE_OR_COPY(Function1) function1,
        ControlQubits... control_qubits)
      {
        return ::ket::mpi::utility::dispatch::diagonal_loop<MpiPolicy>::call(
          parallel_policy, local_state, permutation, communicator, environment,
          target_qubit,
          KET_FORWARD_OR_COPY(Function0, function0),
          KET_FORWARD_OR_COPY(Function1, function1),
          control_qubits...);
      }
#   else // BOOST_NO_CXX11_VARIADIC_TEMPLATES
#     define KET_DIAGONAL_LOOP(z, n, _) \
      template <\
        typename MpiPolicy, typename ParallelPolicy,\
        typename LocalState,\
        typename StateInteger, typename BitInteger, typename Allocator,\
        typename Function0, typename Function1 BOOST_PP_ENUM_TRAILING_PARAMS(n, typename ControlQubit)>\
      inline void diagonal_loop(\
        MpiPolicy const, ParallelPolicy const parallel_policy,\
        LocalState& local_state,\
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator> const& permutation,\
        yampi::communicator const& communicator,\
        yampi::environment const& environment,\
        ::ket::qubit<StateInteger, BitInteger> const target_qubit,\
        KET_RVALUE_REFERENCE_OR_COPY(Function0) function0,\
        KET_RVALUE_REFERENCE_OR_COPY(Function1) function1\
        BOOST_PP_ENUM_TRAILING_BINARY_PARAMS(n, ControlQubit, const control_qubit))\
      {\
        return ::ket::mpi::utility::dispatch::diagonal_loop<MpiPolicy>::call(\
          parallel_policy, local_state, permutation, communicator, environment,\
          target_qubit,\
          KET_FORWARD_OR_COPY(Function0, function0),\
          KET_FORWARD_OR_COPY(Function1, function1)\
          BOOST_PP_ENUM_TRAILING_PARAMS(n, control_qubit));\
      }

      BOOST_PP_REPEAT(KET_MAX_NUM_CONTROL_QUBITS, KET_DIAGONAL_LOOP, _)

#     undef KET_DIAGONAL_LOOP
#   endif // BOOST_NO_CXX11_VARIADIC_TEMPLATES
# endif // KET_USE_DIAGONAL_LOOP
    }
  }
}


# undef KET_RVALUE_REFERENCE_OR_COPY
# undef KET_FORWARD_OR_COPY
# undef KET_array
# undef KET_is_same
# undef KET_is_unsigned
# undef KET_true_type
# undef KET_false_type
# ifdef BOOST_NO_CXX11_STATIC_ASSERT
#   undef static_assert
# endif

#endif

