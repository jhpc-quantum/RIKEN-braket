#ifndef KET_UTILITY_PARALLEL_LOOP_N_HPP
# define KET_UTILITY_PARALLEL_LOOP_N_HPP

# include <boost/config.hpp>

// If you cannot use C++11 and you want to be free from Boost.Thread, define KET_USE_OPENMP and KET_DONT_USE_BOOST_LOCK_GUARD_IN_OPENMP_BLOCKS

# include <cassert>
# include <vector>
# include <iterator>
# include <numeric>
# include <utility>
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
#   include <stdexcept>
#   ifndef BOOST_NO_CXX11_HDR_MUTEX
#     include <mutex>
#   else
#     ifndef KET_DONT_USE_BOOST_LOCK_GUARD_IN_OPENMP_BLOCKS
#       include <boost/thread/lock_guard.hpp>
#     endif
#   endif
#   ifdef __FUJITSU // needed for combination of Boost 1.61.0 and Fujitsu compiler
#     include <boost/utility/in_place_factory.hpp>
#     include <boost/utility/typed_in_place_factory.hpp>
#   endif
#   include <boost/optional.hpp>
# else
#   ifndef BOOST_NO_CXX11_HDR_THREAD
#     include <thread>
#   else
#     include <boost/thread/thread.hpp>
#   endif
# endif
# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   include <type_traits>
# else
#   include <boost/type_traits/integral_constant.hpp>
# endif

# if defined(_OPENMP) && defined(KET_USE_OPENMP)
#   include <omp.h>
# endif

# include <ket/utility/loop_n.hpp>

# ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T&&
#   define KET_FORWARD_OR_COPY(T, x) std::forward<T>(x)
# else
#   define KET_RVALUE_REFERENCE_OR_COPY(T) T
#   define KET_FORWARD_OR_COPY(T, x) x
# endif

# ifndef BOOST_NO_CXX11_HDR_TYPE_TRAITS
#   define KET_true_type std::true_type
#   define KET_false_type std::false_type
# else
#   define KET_true_type boost::true_type
#   define KET_false_type boost::false_type
# endif


namespace ket
{
  namespace utility
  {
    namespace policy
    {
      template <typename NumThreads = int>
      class parallel
      {
        NumThreads num_threads_;

       public:
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
        BOOST_CONSTEXPR parallel() BOOST_NOEXCEPT_OR_NOTHROW
          : num_threads_(static_cast<NumThreads>(omp_get_max_threads()))
        { }

#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
        parallel(NumThreads const num_threads) = delete;
#   else
       private:
        parallel(NumThreads const num_threads);

       public:
#   endif
# else
#   ifndef BOOST_NO_CXX11_HDR_THREAD
        BOOST_CONSTEXPR parallel() BOOST_NOEXCEPT_OR_NOTHROW
          : num_threads_(static_cast<NumThreads>(std::thread::hardware_concurrency()))
        { }
#   else
        BOOST_CONSTEXPR parallel() BOOST_NOEXCEPT_OR_NOTHROW
          : num_threads_(static_cast<NumThreads>(boost::thread::hardware_concurrency()))
        { }
#   endif

        explicit BOOST_CONSTEXPR parallel(NumThreads const num_threads) BOOST_NOEXCEPT_OR_NOTHROW
          : num_threads_(num_threads)
        { }
# endif

        BOOST_CONSTEXPR NumThreads num_threads() const BOOST_NOEXCEPT_OR_NOTHROW
        { return num_threads_; }
      };

      template <typename NumThreads>
      inline BOOST_CONSTEXPR parallel<NumThreads> make_parallel() BOOST_NOEXCEPT_OR_NOTHROW
      { return parallel<NumThreads>(); }

# if defined(_OPENMP) && defined(KET_USE_OPENMP)
#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
      template <typename NumThreads>
      inline parallel<NumThreads>
      make_parallel(NumThreads const num_threads) = delete;
#   endif
# else
      template <typename NumThreads>
      inline BOOST_CONSTEXPR parallel<NumThreads>
      make_parallel(NumThreads const num_threads) BOOST_NOEXCEPT_OR_NOTHROW
      { return parallel<NumThreads>(num_threads); }
# endif

      namespace meta
      {
        template <typename NumThreads>
        struct is_loop_n_policy< ::ket::utility::policy::parallel<NumThreads> >
          : KET_true_type
        { };
      } // namespace meta
    } // namespace policy


    namespace dispatch
    {
      template <typename NumThreads>
      struct num_threads< ::ket::utility::policy::parallel<NumThreads> >
      {
        //static BOOST_CONSTEXPR unsigned int call(
        static unsigned int call(
          ::ket::utility::policy::parallel<NumThreads> const policy) BOOST_NOEXCEPT_OR_NOTHROW
        { return policy.num_threads(); }
      };
    } // namespace dispatch


    namespace dispatch
    {
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
      class omp_mutex
      {
        omp_lock_t omp_lock_;

       public:
        omp_mutex() BOOST_NOEXCEPT_OR_NOTHROW { omp_init_lock(&omp_lock_); }
        ~omp_mutex() BOOST_NOEXCEPT_OR_NOTHROW { omp_destroy_lock(&omp_lock_); }

#   ifndef BOOST_NO_CXX11_DELETED_FUNCTIONS
        omp_mutex(omp_mutex const&) = delete;
        omp_mutex& operator=(omp_mutex const&) = delete;
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        omp_mutex(omp_mutex&&) = delete;
        omp_mutex& operator=(omp_mutex&&) = delete;
#     endif
#   else
       private:
        omp_mutex(omp_mutex const&);
        omp_mutex& operator=(omp_mutex const&);
#     ifndef BOOST_NO_CXX11_RVALUE_REFERENCES
        omp_mutex(omp_mutex&&);
        omp_mutex& operator=(omp_mutex&&);
#     endif

       public:
#   endif

        void lock() BOOST_NOEXCEPT_OR_NOTHROW { omp_set_lock(&omp_lock_); }
        bool try_lock() BOOST_NOEXCEPT_OR_NOTHROW { return static_cast<bool>(omp_test_lock(&omp_lock_)); }
        void unlock() BOOST_NOEXCEPT_OR_NOTHROW { omp_unset_lock(&omp_lock_); }
      };

      class omp_nonstandard_exception
        : public std::runtime_error
      {
       public:
        omp_nonstandard_exception()
          : std::runtime_error("nonstandard exception is thrown in OpenMP block")
        { }
      };

#   if defined(BOOST_NO_CXX11_HDR_THREAD) && defined(KET_DONT_USE_BOOST_LOCK_GUARD_IN_OPENMP_BLOCKS)
      template <typename Mutex>
      class lock_guard
      {
        Mutex& mutex_;

       public:
        typedef Mutex mutex_type;

        lock_guard(Mutex& mutex) BOOST_NOEXCEPT_IF(( BOOST_NOEXCEPT_EXPR(( mutex_.lock() )) ))
          : mutex_(mutex)
        { mutex_.lock(); }

        ~lock_guard() BOOST_NOEXCEPT_IF(( BOOST_NOEXCEPT_EXPR(( mutex_.unlock() )) ))
        { mutex_.unlock(); }
      };
#   endif

      template <typename NumThreads, typename Integer>
      struct loop_n< ::ket::utility::policy::parallel<NumThreads>, Integer>
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

          boost::optional<std::exception> maybe_error;
          bool is_nonstandard_exception_thrown = false;

          ::ket::utility::dispatch::omp_mutex mutex;

#   pragma omp parallel reduction(||:is_nonstandard_exception_thrown)
          {
            int const thread_index = omp_get_thread_num();
#   pragma omp for
            for (Integer count = 0; count < n; ++count)
              try
              {
                function(count, thread_index);
              }
              catch (std::exception& error)
              {
#   ifndef BOOST_NO_CXX11_HDR_THREAD
                std::lock_guard< ::ket::utility::dispatch::omp_mutex> lock(mutex);
#   else
#     ifndef KET_DONT_USE_BOOST_LOCK_GUARD_IN_OPENMP_BLOCKS
                boost::lock_guard< ::ket::utility::dispatch::omp_mutex> lock(mutex);
#     else
                ::ket::utility::dispatch::lock_guard< ::ket::utility::dispatch::omp_mutex> lock(mutex);
#     endif
#   endif

                if (!maybe_error)
                  maybe_error = error;
              }
              catch (...)
              {
                is_nonstandard_exception_thrown = true;
              }
          }

          if (is_nonstandard_exception_thrown)
            throw ::ket::utility::dispatch::omp_nonstandard_exception();

          if (maybe_error)
            throw *maybe_error;
        }
      };
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
#   ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename Function, typename Count>
      struct call_function_from_to
      {
        Function function_;
        Count first_count_;
        Count last_count_;
        int thread_index_;

        call_function_from_to(
          KET_RVALUE_REFERENCE_OR_COPY(Function) function,
          Count const first_count, Count const last_count, int const thread_index)
          : function_(KET_MOVE_OR_COPY(function)),
            first_count_(first_count),
            last_count_(last_count),
            thread_index_(thread_index)
        { }

        void operator()() const
        {
          for (Count count = first_count_; count < last_count_; ++count)
            function_(count, thread_index_);
        }
      };

      template <typename Function, typename Count>
      inline call_function_from_to<Function, Count> make_call_function_from_to(
        Function function, Count const first_count, Count const last_count, int const thrad_index)
      {
        return call_function_from_to<Function, Count>(
          function, first_count, last_count, thread_index);
      }
#   endif

      template <typename NumThreads, typename Integer>
      struct loop_n< ::ket::utility::policy::parallel<NumThreads>, Integer>
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

#   ifndef BOOST_NO_CXX11_HDR_FUTURE
          NumThreads const num_threads = ::ket::utility::num_threads(parallel_policy);
          NumThreads const num_futures = num_threads-1u;
          std::vector<std::future<void> > futures;
          futures.reserve(num_futures);

          NumThreads const local_num_counts = n/num_threads;
          NumThreads const remainder = n%num_threads;
          Integer first_count = 0;

          for (NumThreads thread_index = 0u; thread_index < num_futures; ++thread_index)
          {
            Integer const last_count
              = static_cast<Integer>(
                  local_num_counts*(thread_index+1u) + std::min(remainder, thread_index+1u));

#     ifndef BOOST_NO_CXX11_LAMBDAS
            futures.push_back(std::async(
              std::launch::async,
              [&function, first_count, last_count, thread_index]()
              {
                for (Integer count = first_count; count < last_count; ++count)
                  function(count, static_cast<int>(thread_index));
              }));
#     else // BOOST_NO_CXX11_LAMBDAS
            futures.push_back(std::async(
              std::launch::async,
              make_call_function_from_to(
                function, first_count, last_count, static_cast<int>(thread_index))));
#     endif // BOOST_NO_CXX11_LAMBDAS

            first_count = last_count;
          }

          Integer const last_count
            = static_cast<Integer>(
                local_num_counts*num_threads + std::min(remainder, num_threads));

          for (Integer count = first_count; count < last_count; ++count)
            KET_FORWARD_OR_COPY(function)(count, static_cast<int>(num_futures));

#     ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
          for (std::future<void> const& future: futures)
            futures.wait();
#     else // BOOST_NO_CXX11_RANGE_BASED_FOR
          typedef std::vector<std::future<void> >::const_iterator futures_iterator;

          futures_iterator const last = futures.end();
          for (futures_iterator iter = futures.begin(); iter != last; ++iter)
            iter->wait();
#     endif // BOOST_NO_CXX11_RANGE_BASED_FOR
#   else // BOOST_NO_CXX11_HDR_FUTURE
          NumThreads const num_threads = ::ket::utility::num_threads(parallel_policy);
          NumThreads const num_threads_in_group = num_threads-1u;
          boost::thread_group threads;

          NumThreads const local_num_counts = n/num_threads;
          NumThreads const remainder = n%num_threads;
          Integer first_count = 0;

          for (NumThreads thread_index = 0u; thread_index < num_threads_in_group; ++thread_index)
          {
            Integer const last_count
              = static_cast<Integer>(
                  local_num_counts*(thread_index+1u) + std::min(remainder, thread_index+1u));

#     ifndef BOOST_NO_CXX11_LAMBDAS
            threads.create_thread(
              [&function, first_count, last_count, thread_index]()
              {
                for (Integer count = first_count; count < last_count; ++count)
                  function(count, static_cast<int>(thread_index));
              });
#     else // BOOST_NO_CXX11_LAMBDAS
            threads.create_thread(
              make_call_function_from_to(function, first_count, last_count, static_cast<int>(thread_index)));
#     endif // BOOST_NO_CXX11_LAMBDAS

            first_count = last_count;
          }

          Integer const last_count
            = static_cast<Integer>(
                local_num_counts*num_threads + std::min(remainder, num_threads));

          for (Integer count = first_count; count < last_count; ++count)
            KET_FORWARD_OR_COPY<Function>(function)(count, static_cast<int>(num_threads)-1);

          threads.join_all();
#   endif // BOOST_NO_CXX11_HDR_FUTURE
        }
      };
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)


      // fill
      namespace fill_detail
      {
# ifdef BOOST_NO_CXX11_LAMBDAS
        template <typename ForwardIterator, typename Value>
        struct fill_forward_iterator
        {
          ForwardIterator first_;
          ForwardIterator last_;
          Value const& value_;

          fill_forward_iterator(
            ForwardIterator const first, ForwardIterator const last, Value const& value)
            : first_(first), last_(last), value_(value)
          { }

          typedef void result_type;
          template <typename Difference>
          void operator()(Difference const n, int) const
          {
            ForwardIterator iter = first_;
            std::advance(iter, n);
            *iter = value_;
          }
        };

        template <typename ForwardIterator, typename Value>
        inline ::ket::utility::dispatch::fill_detail::fill_forward_iterator<ForwardIterator, Value>
        make_fill_forward_iterator(
          ForwardIterator const first, ForwardIterator const last, Value const& value)
        {
          return ::ket::utility::dispatch::fill_detail::fill_forward_iterator<ForwardIterator, Value>(
            first, last, value);
        }

        template <typename RandomAccessIterator, typename Value>
        struct fill_random_access_iterator
        {
          RandomAccessIterator first_;
          Value const& value_;

          fill_random_access_iterator(RandomAccessIterator first, Value const& value)
            : first_(first), value_(value)
          { }

          typedef void result_type;
          template <typename Size>
          void operator()(Size const n, int) const { first_[n] = value_; }
        };

        template <typename RandomAccessIterator, typename Value>
        inline
        ::ket::utility::dispatch::fill_detail::fill_random_access_iterator<RandomAccessIterator, Value>
        make_fill_random_access_iterator(RandomAccessIterator first, Value const& value)
        {
          return ::ket::utility::dispatch::fill_detail::fill_random_access_iterator<
            RandomAccessIterator, Value>(
              first, value);
        }
# endif // BOOST_NO_CXX11_LAMBDAS
      } // namespace fill_detail

      template <typename NumThreads>
      struct fill< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator, typename Value>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const& value,
          std::forward_iterator_tag const)
        {
          typedef typename std::iterator_traits<ForwardIterator>::difference_type difference_type;
          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, &value](difference_type const n, int)
            {
              ForwardIterator iter = first;
              std::advance(iter, n);
              *iter = value;
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::dispatch::fill_detail::make_fill_forward_iterator(first, value));
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        template <typename RandomAccessIterator, typename Value>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, Value const& value,
          std::random_access_iterator_tag const)
        {
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef typename std::iterator_traits<RandomAccessIterator>::difference_type difference_type;
          using ::ket::utility::loop_n;
          loop_n(
            parallel_policy, last-first,
            [first, &value](difference_type const n, int) { first[n] = value; });
# else // BOOST_NO_CXX11_LAMBDAS
          using ::ket::utility::loop_n;
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::dispatch::fill_detail::make_fill_random_access_iterator(first, value));
# endif // BOOST_NO_CXX11_LAMBDAS
        }
      };


      // inclusive_scan
      namespace inclusive_scan_detail
      {
# ifdef BOOST_NO_CXX11_LAMBDAS
        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename NumCallsInThreads, typename ItersInThreads, typename OutsInThreads,
          typename PartialSumsInThreads>
        struct inclusive_scan1
        {
          ForwardIterator1 first_;
          ForwardIterator2 d_first_;
          NumCallsInThreads& num_calls_in_threads_;
          ItersInThreads& iters_in_threads_;
          OutsInThreads& outs_in_threads_;
          PartialSumsInThreads& partial_sums_in_threads_;

          inclusive_scan1(
            ForwardIterator1 const first, ForwardIterator2 const d_first,
            NumCallsInThreads& num_calls_in_threads,
            ItersInThreads& iters_in_threads,
            OutsInThreads& outs_in_threads,
            PartialSumsInThreads& partial_sums_in_threads)
            : first_(first), d_first_(d_first),
              num_calls_in_threads_(num_calls_in_threads),
              iters_in_threads_(iters_in_threads),
              outs_in_threads_(outs_in_threads),
              partial_sums_in_threads_(partial_sums_in_threads)
          { }

          typedef void result_type;
          template <typename Size>
          void operator()(Size const n, int const thread_index) const
          {
            if (num_calls_in_threads_[thread_index] == 0)
            {
              iters_in_threads_[thread_index] = first_;
              outs_in_threads_[thread_index] = d_first_;
              std::advance(iters_in_threads_[thread_index], n);
              std::advance(outs_in_threads_[thread_index], n);

              partial_sums_in_threads_[thread_index] = *iters_in_threads_[thread_index]++;
            }
            else
              partial_sums_in_threads_[thread_index]
                = partial_sums_in_threads_[thread_index] + *iters_in_threads_[thread_index]++;

            *outs_in_threads_[thread_index]++ = partial_sums_in_threads_[thread_index];
            ++num_calls_in_threads_[thread_index];
          }
        };

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename NumCallsInThreads, typename ItersInThreads, typename OutsInThreads,
          typename PartialSumsInThreads>
        inline ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan1<
          ForwardIterator1, ForwardIterator2,
          NumCallsInThreads, ItersInThreads, OutsInThreads, PartialSumsInThreads>
        make_inclusive_scan1(
          ForwardIterator1 const first, ForwardIterator2 const d_first,
          NumCallsInThreads& num_calls_in_threads,
          ItersInThreads& iters_in_threads,
          OutsInThreads& outs_in_threads,
          PartialSumsInThreads& partial_sums_in_threads)
        {
          typedef
            ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan1<
              ForwardIterator1, ForwardIterator2,
              NumCallsInThreads, ItersInThreads, OutsInThreads, PartialSumsInThreads>
            result_type;
          return result_type(
            first, d_first,
            num_calls_in_threads, iters_in_threads, outs_in_threads, partial_sums_in_threads);
        }


        template <
          typename ForwardIterator,
          typename NumCallsInThreads, typename OutsInThreads, typename PartialSumsInThreads>
        struct inclusive_scan2
        {
          ForwardIterator d_first_;
          NumCallsInThreads& num_calls_in_threads_;
          OutsInThreads& outs_in_threads_;
          PartialSumsInThreads& partial_sums_in_threads_;

          inclusive_scan2(
            ForwardIterator const d_first,
            NumCallsInThreads& num_calls_in_threads,
            OutsInThreads& outs_in_threads,
            PartialSumsInThreads& partial_sums_in_threads)
            : d_first_(d_first),
              num_calls_in_threads_(num_calls_in_threads),
              outs_in_threads_(outs_in_threads),
              partial_sums_in_threads_(partial_sums_in_threads)
          { }

          typedef void result_type;
          template <typename Size>
          void operator()(Size const n, int const thread_index) const
          {
            if (thread_index == 0)
              return;

            if (num_calls_in_threads_[thread_index] == 0)
            {
              outs_in_threads_[thread_index] = d_first_;
              std::advance(outs_in_threads_[thread_index], n);
            }

            typedef typename std::iterator_traits<ForwardIterator>::value_type value_type;
            value_type const sum
              = partial_sums_in_threads_[thread_index-1] + *outs_in_threads_[thread_index];
            *outs_in_threads_[thread_index]++ = sum;
            ++num_calls_in_threads_[thread_index];
          }
        };

        template <
          typename ForwardIterator,
          typename NumCallsInThreads, typename OutsInThreads, typename PartialSumsInThreads>
        inline ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan2<
          ForwardIterator, NumCallsInThreads, OutsInThreads, PartialSumsInThreads>
        make_inclusive_scan2(
          ForwardIterator const d_first,
          NumCallsInThreads& num_calls_in_threads,
          OutsInThreads& outs_in_threads,
          PartialSumsInThreads& partial_sums_in_threads)
        {
          typedef
            ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan2<
              ForwardIterator, NumCallsInThreads, OutsInThreads, PartialSumsInThreads>
            result_type;
          return result_type(
            d_first, num_calls_in_threads, outs_in_threads, partial_sums_in_threads);
        }


        template <
          typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation,
          typename NumCallsInThreads, typename ItersInThreads, typename OutsInThreads,
          typename PartialSumsInThreads>
        struct inclusive_scan1_
        {
          ForwardIterator1 first_;
          ForwardIterator2 d_first_;
          BinaryOperation binary_operation_;
          NumCallsInThreads& num_calls_in_threads_;
          ItersInThreads& iters_in_threads_;
          OutsInThreads& outs_in_threads_;
          PartialSumsInThreads& partial_sums_in_threads_;

          inclusive_scan1_(
            ForwardIterator1 const first, ForwardIterator2 const d_first,
            BinaryOperation binary_operation,
            NumCallsInThreads& num_calls_in_threads,
            ItersInThreads& iters_in_threads,
            OutsInThreads& outs_in_threads,
            PartialSumsInThreads& partial_sums_in_threads)
            : first_(first), d_first_(d_first), binary_operation_(binary_operation),
              num_calls_in_threads_(num_calls_in_threads),
              iters_in_threads_(iters_in_threads),
              outs_in_threads_(outs_in_threads),
              partial_sums_in_threads_(partial_sums_in_threads)
          { }

          typedef void result_type;
          template <typename Size>
          void operator()(Size const n, int const thread_index) const
          {
            if (num_calls_in_threads_[thread_index] == 0)
            {
              iters_in_threads_[thread_index] = first_;
              outs_in_threads_[thread_index] = d_first_;
              std::advance(iters_in_threads_[thread_index], n);
              std::advance(outs_in_threads_[thread_index], n);

              partial_sums_in_threads_[thread_index] = *iters_in_threads_[thread_index]++;
            }
            else
              partial_sums_in_threads_[thread_index]
                = binary_operation_(
                    partial_sums_in_threads_[thread_index], *iters_in_threads_[thread_index]++);

            *outs_in_threads_[thread_index]++ = partial_sums_in_threads_[thread_index];
            ++num_calls_in_threads_[thread_index];
          }
        };

        template <
          typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation,
          typename NumCallsInThreads, typename ItersInThreads, typename OutsInThreads,
          typename PartialSumsInThreads>
        inline ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan1_<
          ForwardIterator1, ForwardIterator2, BinaryOperation,
          NumCallsInThreads, ItersInThreads, OutsInThreads, PartialSumsInThreads>
        make_inclusive_scan1_(
          ForwardIterator1 const first, ForwardIterator2 const d_first,
          BinaryOperation binary_operation,
          NumCallsInThreads& num_calls_in_threads,
          ItersInThreads& iters_in_threads,
          OutsInThreads& outs_in_threads,
          PartialSumsInThreads& partial_sums_in_threads)
        {
          typedef
            ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan1_<
              ForwardIterator1, ForwardIterator2, BinaryOperation,
              NumCallsInThreads, ItersInThreads, OutsInThreads, PartialSumsInThreads>
            result_type;
          return result_type(
            first, d_first, binary_operation,
            num_calls_in_threads, iters_in_threads, outs_in_threads, partial_sums_in_threads);
        }


        template <
          typename ForwardIterator, typename BinaryOperation,
          typename NumCallsInThreads, typename OutsInThreads, typename PartialSumsInThreads>
        struct inclusive_scan2_
        {
          ForwardIterator d_first_;
          BinaryOperation binary_operation_;
          NumCallsInThreads& num_calls_in_threads_;
          OutsInThreads& outs_in_threads_;
          PartialSumsInThreads& partial_sums_in_threads_;

          inclusive_scan2_(
            ForwardIterator const d_first, BinaryOperation binary_operation,
            NumCallsInThreads& num_calls_in_threads,
            OutsInThreads& outs_in_threads,
            PartialSumsInThreads& partial_sums_in_threads)
            : d_first_(d_first), binary_operation_(binary_operation),
              num_calls_in_threads_(num_calls_in_threads),
              outs_in_threads_(outs_in_threads),
              partial_sums_in_threads_(partial_sums_in_threads)
          { }

          typedef void result_type;
          template <typename Size>
          void operator()(Size const n, int const thread_index) const
          {
            if (thread_index == 0)
              return;

            if (num_calls_in_threads_[thread_index] == 0)
            {
              outs_in_threads_[thread_index] = d_first_;
              std::advance(outs_in_threads_[thread_index], n);
            }

            typedef typename std::iterator_traits<ForwardIterator>::value_type value_type;
            value_type const sum
              = binary_operation_(
                  partial_sums_in_threads_[thread_index-1], *outs_in_threads_[thread_index]);
            *outs_in_threads_[thread_index]++ = sum;
            ++num_calls_in_threads_[thread_index];
          }
        };

        template <
          typename ForwardIterator, typename BinaryOperation,
          typename NumCallsInThreads, typename OutsInThreads, typename PartialSumsInThreads>
        inline ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan2_<
          ForwardIterator, BinaryOperation, NumCallsInThreads, OutsInThreads, PartialSumsInThreads>
        make_inclusive_scan2_(
          ForwardIterator const d_first, BinaryOperation binary_operation,
          NumCallsInThreads& num_calls_in_threads,
          OutsInThreads& outs_in_threads,
          PartialSumsInThreads& partial_sums_in_threads)
        {
          typedef
            ::ket::utility::dispatch::inclusive_scan_detail::inclusive_scan2_<
              ForwardIterator, BinaryOperation,
              NumCallsInThreads, OutsInThreads, PartialSumsInThreads>
            result_type;
          return result_type(
            d_first, binary_operation, num_calls_in_threads, outs_in_threads, partial_sums_in_threads);
        }
# endif // BOOST_NO_CXX11_LAMBDAS
      } // namespace inclusive_scan_detail

      template <typename NumThreads>
      struct inclusive_scan< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator1, typename ForwardIterator2>
        static ForwardIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first)
        {
          std::vector<int> num_calls_in_threads(::ket::utility::num_threads(parallel_policy), 0);
          std::vector<ForwardIterator1> iters_in_threads(::ket::utility::num_threads(parallel_policy));
          std::vector<ForwardIterator2> outs_in_threads(::ket::utility::num_threads(parallel_policy));
          typedef typename std::iterator_traits<ForwardIterator1>::value_type value_type;
          std::vector<value_type> partial_sums_in_threads(::ket::utility::num_threads(parallel_policy));

          typedef typename std::iterator_traits<ForwardIterator1>::difference_type difference_type;
          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, &num_calls_in_threads, &iters_in_threads, &outs_in_threads, &partial_sums_in_threads](
              difference_type const n, int const thread_index)
            {
              if (num_calls_in_threads[thread_index] == 0)
              {
                iters_in_threads[thread_index] = first;
                outs_in_threads[thread_index] = d_first;
                std::advance(iters_in_threads[thread_index], n);
                std::advance(outs_in_threads[thread_index], n);

                partial_sums_in_threads[thread_index] = *iters_in_threads[thread_index]++;
              }
              else
                partial_sums_in_threads[thread_index]
                  = partial_sums_in_threads[thread_index] + *iters_in_threads[thread_index]++;

              *outs_in_threads[thread_index]++ = partial_sums_in_threads[thread_index];
              ++num_calls_in_threads[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::dispatch::inclusive_scan_detail::make_inclusive_scan1(
              first, d_first,
              num_calls_in_threads, iters_in_threads, outs_in_threads, partial_sums_in_threads));
# endif // BOOST_NO_CXX11_LAMBDAS

          std::partial_sum(
            partial_sums_in_threads.begin(), partial_sums_in_threads.end(),
            partial_sums_in_threads.begin());

          std::fill(num_calls_in_threads.begin(), num_calls_in_threads.end(), 0);
# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            [d_first, &num_calls_in_threads, &outs_in_threads, &partial_sums_in_threads](
              difference_type const n, int const thread_index)
            {
              if (thread_index == 0)
                return;

              if (num_calls_in_threads[thread_index] == 0)
              {
                outs_in_threads[thread_index] = d_first;
                std::advance(outs_in_threads[thread_index], n);
              }

              value_type const sum
                = partial_sums_in_threads[thread_index-1] + *outs_in_threads[thread_index];
              *outs_in_threads[thread_index]++ = sum;
              ++num_calls_in_threads[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::dispatch::inclusive_scan_detail::make_inclusive_scan2(
              d_first, num_calls_in_threads, outs_in_threads, partial_sums_in_threads));
# endif // BOOST_NO_CXX11_LAMBDAS

          return outs_in_threads.back();
        }

        template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation>
        static ForwardIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          BinaryOperation binary_operation)
        {
          std::vector<int> num_calls_in_threads(::ket::utility::num_threads(parallel_policy), 0);
          std::vector<ForwardIterator1> iters_in_threads(::ket::utility::num_threads(parallel_policy));
          std::vector<ForwardIterator2> outs_in_threads(::ket::utility::num_threads(parallel_policy));
          typedef typename std::iterator_traits<ForwardIterator1>::value_type value_type;
          std::vector<value_type> partial_sums_in_threads(::ket::utility::num_threads(parallel_policy));

          typedef typename std::iterator_traits<ForwardIterator1>::difference_type difference_type;
          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, &num_calls_in_threads, &iters_in_threads, &outs_in_threads, &partial_sums_in_threads](
              difference_type const n, int const thread_index)
            {
              if (num_calls_in_threads[thread_index] == 0)
              {
                iters_in_threads[thread_index] = first;
                outs_in_threads[thread_index] = d_first;
                std::advance(iters_in_threads[thread_index], n);
                std::advance(outs_in_threads[thread_index], n);

                partial_sums_in_threads[thread_index] = *iters_in_threads[thread_index]++;
              }
              else
                partial_sums_in_threads[thread_index]
                  = binary_operation(
                      partial_sums_in_threads[thread_index], *iters_in_threads[thread_index]++);

              *outs_in_threads[thread_index]++ = partial_sums_in_threads[thread_index];
              ++num_calls_in_threads[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::dispatch::inclusive_scan_detail::make_inclusive_scan1_(
              first, d_first, binary_operation,
              num_calls_in_threads, iters_in_threads, outs_in_threads, partial_sums_in_threads));
# endif // BOOST_NO_CXX11_LAMBDAS

          std::partial_sum(
            partial_sums_in_threads.begin(), partial_sums_in_threads.end(),
            partial_sums_in_threads.begin());

          std::fill(num_calls_in_threads.begin(), num_calls_in_threads.end(), 0);
# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            [d_first, binary_operation, &num_calls_in_threads, &outs_in_threads, &partial_sums_in_threads](
              difference_type const n, int const thread_index)
            {
              if (thread_index != 0)
              {
                if (num_calls_in_threads[thread_index] == 0)
                {
                  outs_in_threads[thread_index] = d_first;
                  std::advance(outs_in_threads[thread_index], n);
                }

                value_type const sum
                  = binary_operation(
                      partial_sums_in_threads[thread_index-1], *outs_in_threads[thread_index]);
                *outs_in_threads[thread_index]++ = sum;
                ++num_calls_in_threads[thread_index];
              }
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::dispatch::inclusive_scan_detail::make_inclusive_scan2_(
              d_first, binary_operation,
              num_calls_in_threads, outs_in_threads, partial_sums_in_threads));
# endif // BOOST_NO_CXX11_LAMBDAS

          return outs_in_threads.back();
        }
      };
    } // namespace dispatch
  } // namespace utility
} // namespace ket


# undef KET_true_type
# undef KET_false_type
# undef KET_RVALUE_REFERENCE_OR_COPY
# undef KET_FORWARD_OR_COPY

#endif

