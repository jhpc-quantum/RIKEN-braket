#ifndef KET_UTILITY_LOOP_N_HPP
# define KET_UTILITY_LOOP_N_HPP

# include <boost/config.hpp>

# include <iterator>
# include <algorithm>
# include <numeric>
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   include <utility>
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
#   include <boost/utility/enable_if.hpp>
# endif

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T&&
#   define KET_FORWARD_OR_COPY(T, x) std::forward<T>(x)
# else
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T
#   define KET_FORWARD_OR_COPY(T, x) x
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_enable_if std::enable_if
#   define KET_true_type std::true_type
#   define KET_false_type std::false_type
# else
#   define KET_enable_if boost::enable_if_c
#   define KET_true_type boost::true_type
#   define KET_false_type boost::false_type
# endif


namespace ket
{
  namespace utility
  {
    namespace policy
    {
      struct sequential { };

      inline BOOST_CONSTEXPR sequential make_sequential() BOOST_NOEXCEPT_OR_NOTHROW
      { return sequential(); }

      namespace meta
      {
        template <typename T>
        struct is_loop_n_policy
          : KET_false_type
        { };

        template <>
        struct is_loop_n_policy< ::ket::utility::policy::sequential >
          : KET_true_type
        { };
      } // namespace meta
    } // namespace policy


    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct num_threads
      {
        static BOOST_CONSTEXPR unsigned int call(ParallelPolicy const);
      };

      template <>
      struct num_threads< ::ket::utility::policy::sequential >
      {
        static BOOST_CONSTEXPR unsigned int call(
          ::ket::utility::policy::sequential const) BOOST_NOEXCEPT_OR_NOTHROW
        { return 1u; }
      };
    } // namespace dispatch

    template <typename ParallelPolicy>
    inline BOOST_CONSTEXPR unsigned int num_threads(ParallelPolicy const policy)
      BOOST_NOEXCEPT_IF(
        BOOST_NOEXCEPT_EXPR(
          ::ket::utility::dispatch::num_threads<ParallelPolicy>::call(policy)))
    { return ::ket::utility::dispatch::num_threads<ParallelPolicy>::call(policy); }


    namespace dispatch
    {
      template <typename ParallelPolicy, typename Integer>
      struct loop_n
      {
        template <typename ParallelPolicy_, typename Function>
        static void call(
          ParallelPolicy_ const, Integer const n,
          KET_RVALUE_REFERENCE_OR_COPY(Function) function);
      };

      template <typename Integer>
      struct loop_n< ::ket::utility::policy::sequential, Integer>
      {
        template <typename ParallelPolicy, typename Function>
        static void call(
          ParallelPolicy const, Integer const n,
          KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
          if (n > 0)
          {
            for (Integer count = 0; count < n-1; ++count)
              function(count, 0);
            std::forward<Function>(function)(n-1, 0);
          }
# else // BOOST_NO_CXX11_RVALUE_REFERENCES
          for (Integer count = 0; count < n; ++count)
            function(count, 0);
# endif // BOOST_NO_CXX11_RVALUE_REFERENCES
        }
      };
    } // namespace dispatch

    template <typename Integer, typename Function>
    inline void loop_n(Integer const n, KET_RVALUE_REFERENCE_OR_COPY(Function) function)
    {
      ::ket::utility::dispatch
      ::loop_n< ::ket::utility::policy::sequential, Integer>::call(
        ::ket::utility::policy::sequential(),
        n, KET_FORWARD_OR_COPY(Function, function));
    }

    template <typename ParallelPolicy, typename Integer, typename Function>
    inline void loop_n(
      ParallelPolicy const parallel_policy,
      Integer const n, KET_RVALUE_REFERENCE_OR_COPY(Function) function)
    {
      ::ket::utility::dispatch::loop_n<ParallelPolicy, Integer>::call(
        parallel_policy, n, KET_FORWARD_OR_COPY(Function, function));
    }


    // fill
    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct fill
      {
        template <typename ForwardIterator, typename Value, typename Category>
        static void call(
          ParallelPolicy const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const& value,
          Category const iterator_category);
      };

      template <>
      struct fill< ::ket::utility::policy::sequential >
      {
        template <typename ForwardIterator, typename Value>
        static void call(
          ::ket::utility::policy::sequential const,
          ForwardIterator const first, ForwardIterator const last, Value const& value,
          std::forward_iterator_tag const)
        { return std::fill(first, last, value); }
      };
    } // namespace dispatch

    template <typename ParallelPolicy, typename ForwardIterator, typename Value>
    inline void fill(
      ParallelPolicy const parallel_policy,
      ForwardIterator const first, ForwardIterator const last, Value const& value)
    {
      ::ket::utility::dispatch::fill<ParallelPolicy>::call(
        parallel_policy, first, last, value,
        typename std::iterator_traits<ForwardIterator>::iterator_category());
    }

    template <typename ForwardIterator, typename Value>
    inline void fill(
      ForwardIterator const first, ForwardIterator const last, Value const& value)
    { ::ket::utility::fill(::ket::utility::policy::make_sequential(), first, last, value); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename ForwardRange, typename Value>
      inline ForwardRange& fill(
        ParallelPolicy const parallel_policy, ForwardRange& range, Value const& value)
      {
        ::ket::utility::fill(parallel_policy, boost::begin(range), boost::end(range), value);
        return range;
      }

      template <typename ForwardRange, typename Value>
      inline ForwardRange& fill(ForwardRange& range, Value const& value)
      {
        ::ket::utility::fill(boost::begin(range), boost::end(range), value);
        return range;
      }
    } // namespace ranges


    // inclusive_scan
    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct inclusive_scan
      {
        template <typename ForwardIterator1, typename ForwardIterator2>
        static ForwardIterator2 call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first);

        template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation>
        static ForwardIterator2 call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          BinaryOperation binary_operation);
      };

      template <>
      struct inclusive_scan< ::ket::utility::policy::sequential >
      {
        template <typename InputIterator, typename OutputIterator>
        static OutputIterator call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last, OutputIterator const d_first)
        { return std::partial_sum(first, last, d_first); }

        template <typename InputIterator, typename OutputIterator, typename BinaryOperation>
        static OutputIterator call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last, OutputIterator const d_first,
          BinaryOperation binary_operation)
        { return std::partial_sum(first, last, d_first, binary_operation); }
      };
    } // namespace dispatch

    template <typename ParallelPolicy, typename ForwardIterator1, typename ForwardIterator2>
    inline typename KET_enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ForwardIterator2>::type
    inclusive_scan(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first)
    {
      return ::ket::utility::dispatch::inclusive_scan<ParallelPolicy>::call(
        parallel_policy, first, last, d_first);
    }

    template <typename InputIterator, typename OutputIterator>
    inline OutputIterator inclusive_scan(
      InputIterator const first, InputIterator const last, OutputIterator const d_first)
    { return std::partial_sum(first, last, d_first); }

    template <
      typename ParallelPolicy,
      typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation>
    inline ForwardIterator2 inclusive_scan(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
      BinaryOperation binary_operation)
    {
      return ::ket::utility::dispatch::inclusive_scan<ParallelPolicy>::call(
        parallel_policy, first, last, d_first, binary_operation);
    }

    template <typename InputIterator, typename OutputIterator, typename BinaryOperation>
    inline typename KET_enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator>::value,
      OutputIterator>::type
    inclusive_scan(
      InputIterator const first, InputIterator const last, OutputIterator d_first,
      BinaryOperation binary_operation)
    { return std::partial_sum(first, last, d_first, binary_operation); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename ForwardRange, typename ForwardIterator>
      inline typename KET_enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
        ForwardIterator>::type
      inclusive_scan(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first)
      {
        return ::ket::utility::inclusive_scan(
          parallel_policy, boost::begin(range), boost::end(range), first);
      }

      template <typename ForwardRange, typename ForwardIterator>
      inline ForwardIterator inclusive_scan(ForwardRange const& range, ForwardIterator const first)
      { return ::ket::utility::inclusive_scan(boost::begin(range), boost::end(range), first); }

      template <
        typename ParallelPolicy,
        typename ForwardRange, typename ForwardIterator, typename BinaryOperation>
      inline ForwardIterator inclusive_scan(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first, BinaryOperation binary_operation)
      {
        return ::ket::utility::inclusive_scan(
          parallel_policy, boost::begin(range), boost::end(range), first, binary_operation);
      }

      template <typename ForwardRange, typename ForwardIterator, typename BinaryOperation>
      inline typename KET_enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value,
        ForwardIterator>::type
      inclusive_scan(
        ForwardRange const& range, ForwardIterator const first, BinaryOperation binary_operation)
      {
        return ::ket::utility::inclusive_scan(
          boost::begin(range), boost::end(range), first, binary_operation);
      }
    } // namespace ranges
  } // namespace utility
} // namespace ket


# undef KET_true_type
# undef KET_false_type
# undef KET_RVALUE_REFERENCE_OR_COPY
# undef KET_FORWARD_OR_COPY

#endif

