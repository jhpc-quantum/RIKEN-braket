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
# ifndef BOOST_NO_CXX11_ADDRESSOF
#   include <memory>
# else
#   include <boost/core/addressof.hpp>
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

# ifndef BOOST_NO_CXX11_ADDRESSOF
#   define KET_addressof std::addressof
# else
#   define KET_addressof boost::addressof
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


    namespace parallel_loop_n_detail
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
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
    } // namespace parallel_loop_n_detail

    namespace dispatch
    {
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
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

          typedef ::ket::utility::parallel_loop_n_detail::omp_mutex mutex_type;
          mutex_type mutex;

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
                typedef std::lock_guard<mutex_type> lock_guard_type;
#   else
#     ifndef KET_DONT_USE_BOOST_LOCK_GUARD_IN_OPENMP_BLOCKS
                typedef boost::lock_guard<mutex_type> lock_guard_type;
#     else
                typedef
                  ::ket::utility::parallel_loop_n_detail::lock_guard<mutex_type>
                  lock_guard_type;
#     endif
#   endif
                lock_guard_type lock(mutex);

                if (!maybe_error)
                  maybe_error = error;
              }
              catch (...)
              {
                is_nonstandard_exception_thrown = true;
              }
          }

          if (is_nonstandard_exception_thrown)
            throw ::ket::utility::parallel_loop_n_detail::omp_nonstandard_exception();

          if (maybe_error)
            throw *maybe_error;
        }
      };
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
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
          std::vector< std::future<void> > futures;
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
              ::ket::utility::parallel_loop_n_detail::make_call_function_from_to(
                function, first_count, last_count, static_cast<int>(thread_index))));
#     endif // BOOST_NO_CXX11_LAMBDAS

            first_count = last_count;
          }

          Integer const last_count
            = static_cast<Integer>(
                local_num_counts*num_threads + std::min(remainder, num_threads));

          for (Integer count = first_count; count < last_count; ++count)
            KET_FORWARD_OR_COPY(Function, function)(count, static_cast<int>(num_futures));

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
              ::ket::utility::parallel_loop_n_detail::make_call_function_from_to(
                function, first_count, last_count, static_cast<int>(thread_index)));
#     endif // BOOST_NO_CXX11_LAMBDAS

            first_count = last_count;
          }

          Integer const last_count
            = static_cast<Integer>(
                local_num_counts*num_threads + std::min(remainder, num_threads));

          for (Integer count = first_count; count < last_count; ++count)
            KET_FORWARD_OR_COPY(Function, function)(count, static_cast<int>(num_threads)-1);

          threads.join_all();
#   endif // BOOST_NO_CXX11_HDR_FUTURE
        }
      };
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
    } // namespace dispatch


    // execute
    namespace parallel_loop_n_detail
    {
# if !defined(_OPENMP) || !defined(KET_USE_OPENMP)
#   ifdef BOOST_NO_CXX11_LAMBDAS
      template <typename Function>
      struct call_execute
      {
        Function function_;
        int thread_index_;

        call_execute(
          KET_RVALUE_REFERENCE_OR_COPY(Function) function, int const thread_index)
          : function_(KET_MOVE_OR_COPY(function)),
            thread_index_(thread_index)
        { }

        void operator()() const { function_(thread_index_); }
      };

      template <typename Function>
      inline call_execute<Function> make_call_execute(
        Function function, int const thrad_index)
      { return call_execute<Function>(function, thread_index); }
#   endif
# endif // !defined(_OPENMP) || !defined(KET_USE_OPENMP)
    } // namespace parallel_loop_n_detail

    namespace dispatch
    {
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
      template <typename NumThreads>
      struct execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

          boost::optional<std::exception> maybe_error;
          bool is_nonstandard_exception_thrown = false;

          typedef ::ket::utility::parallel_loop_n_detail::omp_mutex mutex_type;
          mutex_type mutex;

#   pragma omp parallel reduction(||:is_nonstandard_exception_thrown)
          {
            try
            {
              function(omp_get_thread_num());
            }
            catch (std::exception& error)
            {
#   ifndef BOOST_NO_CXX11_HDR_THREAD
              typedef std::lock_guard<mutex_type> lock_guard_type;
#   else
#     ifndef KET_DONT_USE_BOOST_LOCK_GUARD_IN_OPENMP_BLOCKS
              typedef boost::lock_guard<mutex_type> lock_guard_type;
#     else
              typedef
                ::ket::utility::parallel_loop_n_detail::lock_guard<mutex_type>
                lock_guard_type;
#     endif
#   endif
              lock_guard_type lock(mutex);

              if (!maybe_error)
                maybe_error = error;
            }
            catch (...)
            {
              is_nonstandard_exception_thrown = true;
            }
          }

          if (is_nonstandard_exception_thrown)
            throw ::ket::utility::parallel_loop_n_detail::omp_nonstandard_exception();

          if (maybe_error)
            throw *maybe_error;
        }
      };
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
      template <typename NumThreads>
      struct execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          KET_RVALUE_REFERENCE_OR_COPY(Function) function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

#   ifndef BOOST_NO_CXX11_HDR_FUTURE
          NumThreads const num_threads = ::ket::utility::num_threads(parallel_policy);
          NumThreads const num_futures = num_threads-1u;
          std::vector< std::future<void> > futures;
          futures.reserve(num_futures);

          for (NumThreads thread_index = 0u; thread_index < num_futures; ++thread_index)
          {
#     ifndef BOOST_NO_CXX11_LAMBDAS
            futures.push_back(std::async(
              std::launch::async,
              [&function, thread_index]()
              { function(static_cast<int>(thread_index)); }));
#     else // BOOST_NO_CXX11_LAMBDAS
            futures.push_back(std::async(
              std::launch::async,
              ::ket::utility::parallel_loop_n_detail::make_call_execute(
                function, static_cast<int>(thread_index))));
#     endif // BOOST_NO_CXX11_LAMBDAS
          }

          KET_FORWARD_OR_COPY(Function, function)(static_cast<int>(num_futures));

#     ifndef BOOST_NO_CXX11_RANGE_BASED_FOR
          for (std::future<void> const& future: futures)
            futures.wait();
#     else // BOOST_NO_CXX11_RANGE_BASED_FOR
          typedef std::vector< std::future<void> >::const_iterator futures_iterator;

          futures_iterator const last = futures.end();
          for (futures_iterator iter = futures.begin(); iter != last; ++iter)
            iter->wait();
#     endif // BOOST_NO_CXX11_RANGE_BASED_FOR
#   else // BOOST_NO_CXX11_HDR_FUTURE
          NumThreads const num_threads = ::ket::utility::num_threads(parallel_policy);
          NumThreads const num_threads_in_group = num_threads-1u;
          boost::thread_group threads;

          for (NumThreads thread_index = 0u; thread_index < num_threads_in_group; ++thread_index)
          {
#     ifndef BOOST_NO_CXX11_LAMBDAS
            threads.create_thread(
              [&function, thread_index]()
              { function(static_cast<int>(thread_index)); });
#     else // BOOST_NO_CXX11_LAMBDAS
            threads.create_thread(
              ::ket::utility::parallel_loop_n_detail::make_call_function_from_to(
                function, static_cast<int>(thread_index)));
#     endif // BOOST_NO_CXX11_LAMBDAS
          }

          KET_FORWARD_OR_COPY(Function, function)(static_cast<int>(num_threads)-1);

          threads.join_all();
#   endif // BOOST_NO_CXX11_HDR_FUTURE
        }
      };
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
    } // namespace dispatch


    // fill
    namespace parallel_loop_n_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename ForwardIterator, typename Value,
        typename IsCalledsPtr, typename ItersPtr>
      struct fill_forward_iterator
      {
        ForwardIterator first_;
        Value const& value_;
        IsCalledsPtr is_calleds_ptr_;
        ItersPtr iters_ptr_;

        fill_forward_iterator(
          ForwardIterator const first, Value const& value,
          IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr)
          : first_(first), value_(value),
            is_calleds_ptr_(is_calleds_ptr), iters_ptr_(iters_ptr)
        { }

        typedef void result_type;
        template <typename Difference>
        void operator()(Difference const n, int thread_index) const
        {
          if (not static_cast<bool>(is_calleds_ptr_[thread_index]))
          {
            iters_ptr_[thread_index] = first_;
            std::advance(iters_ptr_[thread_index], n);
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }
          *iters_ptr_[thread_index]++ = value_;
        }
      };

      template <
        typename ForwardIterator, typename Value,
        typename IsCalledsPtr, typename ItersPtr>
      inline ::ket::utility::parallel_loop_n_detail::fill_forward_iterator<ForwardIterator, Value, IsCalledsPtr, ItersPtr>
      make_fill_forward_iterator(
        ForwardIterator const first, Value const& value,
        IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr)
      {
        return ::ket::utility::parallel_loop_n_detail::fill_forward_iterator<ForwardIterator, Value, IsCalledsPtr, ItersPtr>(
          first, value, is_calleds_ptr, iters_ptr);
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
      ::ket::utility::parallel_loop_n_detail::fill_random_access_iterator<RandomAccessIterator, Value>
      make_fill_random_access_iterator(RandomAccessIterator first, Value const& value)
      {
        return ::ket::utility::parallel_loop_n_detail::fill_random_access_iterator<
          RandomAccessIterator, Value>(
            first, value);
      }
# endif // BOOST_NO_CXX11_LAMBDAS
    } // namespace parallel_loop_n_detail

    namespace dispatch
    {
      template <typename NumThreads>
      struct fill< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator, typename Value>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const& value,
          std::forward_iterator_tag const)
        {
          std::vector<int> is_calleds(
            ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          std::vector<ForwardIterator> iters(
            ::ket::utility::num_threads(parallel_policy));
          int* is_calleds_ptr = KET_addressof(is_calleds.front());
          ForwardIterator* iters_ptr = KET_addressof(iters.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<ForwardIterator>::difference_type
            difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, &value, is_calleds_ptr, iters_ptr](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds_ptr[thread_index]))
              {
                iters_ptr[thread_index] = first;
                std::advance(iters_ptr[thread_index], n);
                is_calleds_ptr[thread_index] = static_cast<int>(true);
              }
              *iters_ptr[thread_index]++ = value;
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::parallel_loop_n_detail::make_fill_forward_iterator(
              first, value, is_calleds_ptr, iters_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS
        }

        template <typename RandomAccessIterator, typename Value>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, Value const& value,
          std::random_access_iterator_tag const)
        {
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<RandomAccessIterator>::difference_type
            difference_type;
          using ::ket::utility::loop_n;
          loop_n(
            parallel_policy, last-first,
            [first, &value](difference_type const n, int) { first[n] = value; });
# else // BOOST_NO_CXX11_LAMBDAS
          using ::ket::utility::loop_n;
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::parallel_loop_n_detail::make_fill_random_access_iterator(
              first, value));
# endif // BOOST_NO_CXX11_LAMBDAS
        }
      };
    } // namespace dispatch


    // inclusive_scan
    namespace parallel_loop_n_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      struct inclusive_scan_forward_iterator
      {
        ForwardIterator1 first_;
        ForwardIterator2 d_first_;
        IsCalledsPtr is_calleds_ptr_;
        ItersPtr iters_ptr_;
        OutsPtr outs_ptr_;
        PartialSumsPtr partial_sums_ptr_;

        inclusive_scan_forward_iterator(
          ForwardIterator1 const first, ForwardIterator2 const d_first,
          IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
          PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first),
            is_calleds_ptr_(is_calleds_ptr), iters_ptr_(iters_ptr),
            outs_ptr_(outs_ptr), partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (not is_calleds_ptr_[thread_index])
          {
            iters_ptr_[thread_index] = first_;
            outs_ptr_[thread_index] = d_first_;
            std::advance(iters_ptr_[thread_index], n);
            std::advance(outs_ptr_[thread_index], n);

            partial_sums_ptr_[thread_index] = *iters_ptr_[thread_index]++;
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }
          else
            partial_sums_ptr_[thread_index] += *iters_ptr_[thread_index]++;

          *outs_ptr_[thread_index]++ = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::inclusive_scan_forward_iterator<
        ForwardIterator1, ForwardIterator2,
        IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
      make_inclusive_scan_forward_iterator(
        ForwardIterator1 const first, ForwardIterator2 const d_first,
        IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::inclusive_scan_forward_iterator<
            ForwardIterator1, ForwardIterator2,
            IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
          result_type;
        return result_type(
          first, d_first, is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr);
      }


      template <
        typename ForwardIterator,
        typename IsCalledsPtr, typename OutsPtr, typename PartialSumsPtr>
      struct post_inclusive_scan_forward_iterator
      {
        ForwardIterator d_first_;
        IsCalledsPtr is_calleds_ptr_;
        OutsPtr outs_ptr_;
        PartialSumsPtr partial_sums_ptr_;

        post_inclusive_scan_forward_iterator(
          ForwardIterator const d_first,
          IsCalledsPtr is_calleds_ptr,
          OutsPtr outs_ptr,
          PartialSumsPtr partial_sums_ptr)
          : d_first_(d_first),
            is_calleds_ptr_(is_calleds_ptr), outs_ptr_(outs_ptr),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (thread_index == 0)
            return;

          if (not is_calleds_ptr_[thread_index])
          {
            outs_ptr_[thread_index] = d_first_;
            std::advance(outs_ptr_[thread_index], n);
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }

          *outs_ptr_[thread_index]++ += partial_sums_ptr_[thread_index-1];
        }
      };

      template <
        typename ForwardIterator,
        typename IsCalledsPtr, typename OutsPtr, typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_forward_iterator<
        ForwardIterator, IsCalledsPtr, OutsPtr, PartialSumsPtr>
      make_post_inclusive_scan_forward_iterator(
        ForwardIterator const d_first,
        IsCalledsPtr is_calleds_ptr, OutsPtr outs_ptr, PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_forward_iterator<
            ForwardIterator, IsCalledsPtr, OutsPtr, PartialSumsPtr>
          result_type;
        return result_type(d_first, is_calleds_ptr, outs_ptr, partial_sums_ptr);
      }


      template <
        typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename PartialSumsPtr>
      struct inclusive_scan_random_access_iterator
      {
        RandomAccessIterator1 first_;
        RandomAccessIterator2 d_first_;
        PartialSumsPtr partial_sums_ptr_;

        inclusive_scan_random_access_iterator(
          RandomAccessIterator1 const first, RandomAccessIterator2 const d_first,
          PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          partial_sums_ptr_[thread_index] += first_[n];
          d_first_[n] = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::inclusive_scan_random_access_iterator<
        RandomAccessIterator1, RandomAccessIterator2, PartialSumsPtr>
      make_inclusive_scan_random_access_iterator(
        RandomAccessIterator1 const first, RandomAccessIterator2 const d_first,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::inclusive_scan_random_access_iterator<
            RandomAccessIterator1, RandomAccessIterator2, PartialSumsPtr>
          result_type;
        return result_type(first, d_first, partial_sums_ptr);
      }


      template <typename RandomAccessIterator, typename PartialSumsPtr>
      struct post_inclusive_scan_random_access_iterator
      {
        RandomAccessIterator d_first_;
        PartialSumsPtr partial_sums_ptr_;

        post_inclusive_scan_random_access_iterator(
          RandomAccessIterator const d_first, PartialSumsPtr partial_sums_ptr)
          : d_first_(d_first), partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (thread_index == 0)
            return;

          d_first_[n] += partial_sums_ptr_[thread_index-1];
        }
      };

      template <typename RandomAccessIterator, typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_random_access_iterator<
        RandomAccessIterator, PartialSumsPtr>
      make_post_inclusive_scan_random_access_iterator(
        RandomAccessIterator const d_first, PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_random_access_iterator<
            RandomAccessIterator, PartialSumsPtr>
          result_type;
        return result_type(d_first, partial_sums_ptr);
      }


      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      struct inclusive_scan_forward_iterator_
      {
        ForwardIterator1 first_;
        ForwardIterator2 d_first_;
        BinaryOperation binary_operation_;
        IsCalledsPtr is_calleds_ptr_;
        ItersPtr iters_ptr_;
        OutsPtr outs_ptr_;
        PartialSumsPtr partial_sums_ptr_;

        inclusive_scan_forward_iterator_(
          ForwardIterator1 const first, ForwardIterator2 const d_first,
          BinaryOperation binary_operation,
          IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
          PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first), binary_operation_(binary_operation),
            is_calleds_ptr_(is_calleds_ptr),
            iters_ptr_(iters_ptr), outs_ptr_(outs_ptr),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (not is_calleds_ptr_[thread_index])
          {
            iters_ptr_[thread_index] = first_;
            outs_ptr_[thread_index] = d_first_;
            std::advance(iters_ptr_[thread_index], n);
            std::advance(outs_ptr_[thread_index], n);

            partial_sums_ptr_[thread_index] = *iters_ptr_[thread_index]++;
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }
          else
            partial_sums_ptr_[thread_index]
              = binary_operation_(
                  partial_sums_ptr_[thread_index], *iters_ptr_[thread_index]++);

          *outs_ptr_[thread_index]++ = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::inclusive_scan_forward_iterator_<
        ForwardIterator1, ForwardIterator2, BinaryOperation,
        IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
      make_inclusive_scan_forward_iterator_(
        ForwardIterator1 const first, ForwardIterator2 const d_first,
        BinaryOperation binary_operation,
        IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::inclusive_scan_forward_iterator_<
            ForwardIterator1, ForwardIterator2, BinaryOperation,
            IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
          result_type;
        return result_type(
          first, d_first, binary_operation,
          is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr);
      }


      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation, typename Value,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      struct inclusive_scan_forward_iterator_init
      {
        ForwardIterator1 first_;
        ForwardIterator2 d_first_;
        BinaryOperation binary_operation_;
        Value initial_value_;
        IsCalledsPtr is_calleds_ptr_;
        ItersPtr iters_ptr_;
        OutsPtr outs_ptr_;
        PartialSumsPtr partial_sums_ptr_;

        inclusive_scan_forward_iterator_init(
          ForwardIterator1 const first, ForwardIterator2 const d_first,
          BinaryOperation binary_operation, Value const initial_value,
          IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
          PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first), binary_operation_(binary_operation),
            initial_value_(initial_value),
            is_calleds_ptr_(is_calleds_ptr),
            iters_ptr_(iters_ptr), outs_ptr_(outs_ptr),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (not is_calleds_ptr_[thread_index])
          {
            iters_ptr_[thread_index] = first_;
            outs_ptr_[thread_index] = d_first_;
            std::advance(iters_ptr_[thread_index], n);
            std::advance(outs_ptr_[thread_index], n);

            partial_sums_ptr_[thread_index]
              = thread_index == 0
                ? binary_operation_(initial_value_, *iters_ptr_[0]++)
                : *iters_ptr_[thread_index]++;
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }
          else
            partial_sums_ptr_[thread_index]
              = binary_operation_(
                  partial_sums_ptr_[thread_index], *iters_ptr_[thread_index]++);

          *outs_ptr_[thread_index]++ = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation, typename Value,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::inclusive_scan_forward_iterator_init<
        ForwardIterator1, ForwardIterator2, BinaryOperation, Value,
        IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
      make_inclusive_scan_forward_iterator_init(
        ForwardIterator1 const first, ForwardIterator2 const d_first,
        BinaryOperation binary_operation, Value const initial_value,
        IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::inclusive_scan_forward_iterator_init<
            ForwardIterator1, ForwardIterator2, BinaryOperation, Value,
            IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
          result_type;
        return result_type(
          first, d_first, binary_operation, initial_value,
          is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr);
      }


      template <
        typename ForwardIterator, typename BinaryOperation,
        typename IsCalledsPtr, typename OutsPtr, typename PartialSumsPtr>
      struct post_inclusive_scan_forward_iterator_
      {
        ForwardIterator d_first_;
        BinaryOperation binary_operation_;
        IsCalledsPtr is_calleds_ptr_;
        OutsPtr outs_ptr_;
        PartialSumsPtr partial_sums_ptr_;

        post_inclusive_scan_forward_iterator_(
          ForwardIterator const d_first, BinaryOperation binary_operation,
          IsCalledsPtr is_calleds_ptr, OutsPtr outs_ptr,
          PartialSumsPtr partial_sums_ptr)
          : d_first_(d_first), binary_operation_(binary_operation),
            is_calleds_ptr_(is_calleds_ptr), outs_ptr_(outs_ptr),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (thread_index == 0)
            return;

          if (not is_calleds_ptr_[thread_index])
          {
            outs_ptr_[thread_index] = d_first_;
            std::advance(outs_ptr_[thread_index], n);
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }

          *outs_ptr_[thread_index]++
            = binary_operation_(
                partial_sums_ptr_[thread_index-1], *outs_ptr_[thread_index]);
        }
      };

      template <
        typename ForwardIterator, typename BinaryOperation,
        typename IsCalledsPtr, typename OutsPtr, typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_forward_iterator_<
        ForwardIterator, BinaryOperation, IsCalledsPtr, OutsPtr, PartialSumsPtr>
      make_post_inclusive_scan_forward_iterator_(
        ForwardIterator const d_first, BinaryOperation binary_operation,
        IsCalledsPtr is_calleds_ptr, OutsPtr outs_ptr, PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_forward_iterator_<
            ForwardIterator, BinaryOperation, IsCalledsPtr, OutsPtr, PartialSumsPtr>
          result_type;
        return result_type(
          d_first, binary_operation, is_calleds_ptr, outs_ptr, partial_sums_ptr);
      }


      template <
        typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename BinaryOperation, typename PartialSumsPtr>
      struct inclusive_scan_random_access_iterator_
      {
        RandomAccessIterator1 first_;
        RandomAccessIterator2 d_first_;
        BinaryOperation binary_operation_;
        PartialSumsPtr partial_sums_ptr_;

        inclusive_scan_random_access_iterator_(
          RandomAccessIterator1 const first, RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first), binary_operation_(binary_operation),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          partial_sums_ptr_[thread_index]
            = binary_operation_(partial_sums_ptr_[thread_index], first_[n]);
          d_first_[n] = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename BinaryOperation, typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::inclusive_scan_random_access_iterator_<
        RandomAccessIterator1, RandomAccessIterator2, BinaryOperation, PartialSumsPtr>
      make_inclusive_scan_random_access_iterator_(
        RandomAccessIterator1 const first, RandomAccessIterator2 const d_first,
        BinaryOperation binary_operation, PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::inclusive_scan_random_access_iterator_<
            RandomAccessIterator1, RandomAccessIterator2, BinaryOperation,
            PartialSumsPtr>
          result_type;
        return result_type(first, d_first, binary_operation, partial_sums_ptr);
      }


      template <
        typename RandomAccessIterator, typename BinaryOperation,
        typename PartialSumsPtr>
      struct post_inclusive_scan_random_access_iterator_
      {
        RandomAccessIterator d_first_;
        BinaryOperation binary_operation_;
        PartialSumsPtr partial_sums_ptr_;

        post_inclusive_scan_random_access_iterator_(
          RandomAccessIterator const d_first, BinaryOperation binary_operation,
          PartialSumsPtr partial_sums_ptr)
          : d_first_(d_first), binary_operation_(binary_operation),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (thread_index == 0)
            return;

          d_first_[n]
            = binary_operation_(partial_sums_ptr_[thread_index-1], d_first_[n]);
        }
      };

      template <
        typename RandomAccessIterator, typename BinaryOperation,
        typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_random_access_iterator_<
        RandomAccessIterator, BinaryOperation, PartialSumsPtr>
      make_post_inclusive_scan_random_access_iterator_(
        RandomAccessIterator const d_first, BinaryOperation binary_operation,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan_random_access_iterator_<
            RandomAccessIterator, BinaryOperation, PartialSumsPtr>
          result_type;
        return result_type(d_first, binary_operation, partial_sums_ptr);
      }
# endif // BOOST_NO_CXX11_LAMBDAS


      template <
        typename ParallelPolicy,
        typename RangeSize, typename ForwardIterator, typename BinaryOperation,
        typename Value, typename NumThreads>
      static void post_inclusive_scan(
        ParallelPolicy const parallel_policy,
        RangeSize const range_size, ForwardIterator d_first, BinaryOperation binary_operation,
        int* is_calleds_ptr, Value* partial_sums_ptr, ForwardIterator* outs_ptr, NumThreads const num_threads)
      {
        std::partial_sum(
          partial_sums_ptr, partial_sums_ptr + num_threads,
          partial_sums_ptr, binary_operation);

        std::fill(
          is_calleds_ptr, is_calleds_ptr + num_threads,
          static_cast<int>(false));

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy, range_size,
          [d_first, binary_operation, is_calleds_ptr, partial_sums_ptr, outs_ptr](
            RangeSize const n, int const thread_index)
          {
            if (thread_index == 0)
              return;

            if (not is_calleds_ptr[thread_index])
            {
              outs_ptr[thread_index] = d_first;
              std::advance(outs_ptr[thread_index], n);
              is_calleds_ptr[thread_index] = static_cast<int>(true);
            }

            *outs_ptr[thread_index]++
              = binary_operation(
                  partial_sums_ptr[thread_index-1], *outs_ptr[thread_index]);
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy, range_size,
          ::ket::utility::parallel_loop_n_detail::make_post_inclusive_scan_forward_iterator_(
            d_first, binary_operation, is_calleds_ptr, outs_ptr, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS
      }

      template <
        typename ParallelPolicy,
        typename RangeSize, typename RandomAccessIterator, typename BinaryOperation,
        typename Value, typename NumThreads>
      static void post_inclusive_scan(
        ParallelPolicy const parallel_policy,
        RangeSize const range_size, RandomAccessIterator d_first, BinaryOperation binary_operation,
        Value* partial_sums_ptr, NumThreads const num_threads)
      {
        std::partial_sum(
          partial_sums_ptr, partial_sums_ptr + num_threads,
          partial_sums_ptr, binary_operation);

        using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy, range_size,
          [d_first, binary_operation, partial_sums_ptr](
            RangeSize const n, int const thread_index)
          {
            if (thread_index == 0)
              return;

            d_first[n]
              = binary_operation(partial_sums_ptr[thread_index-1], d_first[n]);
          });
# else // BOOST_NO_CXX11_LAMBDAS
        loop_n(
          parallel_policy, range_size,
          ::ket::utility::parallel_loop_n_detail::make_post_inclusive_scan_random_access_iterator_(
            d_first, binary_operation, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS
      }
    } // namespace parallel_loop_n_detail

    namespace dispatch
    {
      template <typename NumThreads>
      struct inclusive_scan< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator1, typename ForwardIterator2>
        static ForwardIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        {
          std::vector<int> is_calleds(
            ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          std::vector<ForwardIterator1> iters(
            ::ket::utility::num_threads(parallel_policy));
          std::vector<ForwardIterator2> outs(
            ::ket::utility::num_threads(parallel_policy));
          typedef
            typename std::iterator_traits<ForwardIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy));

          int* is_calleds_ptr = KET_addressof(is_calleds.front());
          ForwardIterator1* iters_ptr = KET_addressof(iters.front());
          ForwardIterator2* outs_ptr = KET_addressof(outs.front());
          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<ForwardIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_ptr[thread_index])
              {
                iters_ptr[thread_index] = first;
                outs_ptr[thread_index] = d_first;
                std::advance(iters_ptr[thread_index], n);
                std::advance(outs_ptr[thread_index], n);

                partial_sums_ptr[thread_index] = *iters_ptr[thread_index]++;
                is_calleds_ptr[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_ptr[thread_index] += *iters_ptr[thread_index]++;

              *outs_ptr[thread_index]++ = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::parallel_loop_n_detail::make_inclusive_scan_forward_iterator(
              first, d_first,
              is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          std::partial_sum(
            partial_sums_ptr, partial_sums_ptr + partial_sums.size(),
            partial_sums_ptr);

          std::fill(
            is_calleds_ptr, is_calleds_ptr + is_calleds.size(),
            static_cast<int>(false));
# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            [d_first, is_calleds_ptr, outs_ptr, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              if (thread_index == 0)
                return;

              if (not is_calleds_ptr[thread_index])
              {
                outs_ptr[thread_index] = d_first;
                std::advance(outs_ptr[thread_index], n);
                is_calleds_ptr[thread_index] = static_cast<int>(true);
              }

              *outs_ptr[thread_index]++ += partial_sums_ptr[thread_index-1];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::parallel_loop_n_detail::make_post_inclusive_scan_forward_iterator(
              d_first, is_calleds_ptr, outs_ptr, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          return outs.back();
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2>
        static RandomAccessIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last,
          RandomAccessIterator2 const d_first,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy), static_cast<value_type>(0));

          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, last-first,
            [first, d_first, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              partial_sums_ptr[thread_index] += first[n];
              d_first[n] = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::parallel_loop_n_detail::make_inclusive_scan_random_access_iterator(
              first, d_first, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          std::partial_sum(
            partial_sums_ptr, partial_sums_ptr + partial_sums.size(),
            partial_sums_ptr);

# ifndef BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, last-first,
            [d_first, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              if (thread_index == 0)
                return;

              d_first[n] += partial_sums_ptr[thread_index-1];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::parallel_loop_n_detail::make_post_inclusive_scan_random_access_iterator(
              d_first, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          return d_first + (last-first);
        }

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation>
        static ForwardIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          BinaryOperation binary_operation,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        {
          std::vector<int> is_calleds(
            ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          std::vector<ForwardIterator1> iters(
            ::ket::utility::num_threads(parallel_policy));
          std::vector<ForwardIterator2> outs(
            ::ket::utility::num_threads(parallel_policy));
          typedef
            typename std::iterator_traits<ForwardIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy));

          int* is_calleds_ptr = KET_addressof(is_calleds.front());
          ForwardIterator1* iters_ptr = KET_addressof(iters.front());
          ForwardIterator2* outs_ptr = KET_addressof(outs.front());
          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<ForwardIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation,
             is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_ptr[thread_index])
              {
                iters_ptr[thread_index] = first;
                outs_ptr[thread_index] = d_first;
                std::advance(iters_ptr[thread_index], n);
                std::advance(outs_ptr[thread_index], n);

                partial_sums_ptr[thread_index] = *iters_ptr[thread_index]++;
                is_calleds_ptr[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_ptr[thread_index]
                  = binary_operation(
                      partial_sums_ptr[thread_index], *iters_ptr[thread_index]++);

              *outs_ptr[thread_index]++ = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::parallel_loop_n_detail::make_inclusive_scan_forward_iterator_(
              first, d_first, binary_operation,
              is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds_ptr, partial_sums_ptr, outs_ptr, partial_sums.size());

          return outs.back();
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename BinaryOperation>
        static RandomAccessIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last,
          RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy), static_cast<value_type>(0));

          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, last-first,
            [first, d_first, binary_operation, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              partial_sums_ptr[thread_index]
                = binary_operation(partial_sums_ptr[thread_index], first[n]);
              d_first[n] = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::parallel_loop_n_detail::make_inclusive_scan_random_access_iterator_(
              first, d_first, binary_operation, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last-first, d_first, binary_operation,
            partial_sums_ptr, partial_sums.size());

          return d_first + (last-first);
        }

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation, typename Value>
        static ForwardIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          BinaryOperation binary_operation, Value const initial_value,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        {
          std::vector<int> is_calleds(
            ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          std::vector<ForwardIterator1> iters(
            ::ket::utility::num_threads(parallel_policy));
          std::vector<ForwardIterator2> outs(
            ::ket::utility::num_threads(parallel_policy));
          typedef
            typename std::iterator_traits<ForwardIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy));

          int* is_calleds_ptr = KET_addressof(is_calleds.front());
          ForwardIterator1* iters_ptr = KET_addressof(iters.front());
          ForwardIterator2* outs_ptr = KET_addressof(outs.front());
          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<ForwardIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, initial_value,
             is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_ptr[thread_index])
              {
                iters_ptr[thread_index] = first;
                outs_ptr[thread_index] = d_first;
                std::advance(iters_ptr[thread_index], n);
                std::advance(outs_ptr[thread_index], n);

                partial_sums_ptr[thread_index]
                  = thread_index == 0
                    ? binary_operation(initial_value, *iters_ptr[0]++)
                    : *iters_ptr[thread_index]++;
                is_calleds_ptr[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_ptr[thread_index]
                  = binary_operation(
                      partial_sums_ptr[thread_index], *iters_ptr[thread_index]++);

              *outs_ptr[thread_index]++ = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::parallel_loop_n_detail::make_inclusive_scan_forward_iterator_init(
              first, d_first, binary_operation, initial_value,
              is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds_ptr, partial_sums_ptr, outs_ptr, partial_sums.size());

          return outs.back();
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename BinaryOperation, typename Value>
        static RandomAccessIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last,
          RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, Value const initial_value,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy), static_cast<value_type>(0));
          partial_sums.front() = initial_value;

          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, last-first,
            [first, d_first, binary_operation, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              partial_sums_ptr[thread_index]
                = binary_operation(partial_sums_ptr[thread_index], first[n]);
              d_first[n] = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::parallel_loop_n_detail::make_inclusive_scan_random_access_iterator_(
              first, d_first, binary_operation, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last-first, d_first, binary_operation,
            partial_sums_ptr, partial_sums.size());

          return d_first + (last-first);
        }
      };
    } // namespace dispatch


    // transform inclusive_scan
    namespace parallel_loop_n_detail
    {
# ifdef BOOST_NO_CXX11_LAMBDAS
      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation, typename UnaryOperation,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      struct transform_inclusive_scan_forward_iterator
      {
        ForwardIterator1 first_;
        ForwardIterator2 d_first_;
        BinaryOperation binary_operation_;
        UnaryOperation unary_operation_;
        IsCalledsPtr is_calleds_ptr_;
        ItersPtr iters_ptr_;
        OutsPtr outs_ptr_;
        PartialSumsPtr partial_sums_ptr_;

        transform_inclusive_scan_forward_iterator(
          ForwardIterator1 const first, ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
          PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first),
            binary_operation_(binary_operation),
            unary_operation_(unary_operation),
            is_calleds_ptr_(is_calleds_ptr),
            iters_ptr_(iters_ptr), outs_ptr_(outs_ptr),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (not is_calleds_ptr_[thread_index])
          {
            iters_ptr_[thread_index] = first_;
            outs_ptr_[thread_index] = d_first_;
            std::advance(iters_ptr_[thread_index], n);
            std::advance(outs_ptr_[thread_index], n);

            partial_sums_ptr_[thread_index]
              = unary_operation_(*iters_ptr_[thread_index]++);
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }
          else
            partial_sums_ptr_[thread_index]
              = binary_operation_(
                  partial_sums_ptr_[thread_index],
                  unary_operation_(*iters_ptr_[thread_index]++));

          *outs_ptr_[thread_index]++ = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation, typename UnaryOperation,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::transform_inclusive_scan_forward_iterator<
        ForwardIterator1, ForwardIterator2, BinaryOperation, UnaryOperation,
        IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
      make_transform_inclusive_scan_forward_iterator(
        ForwardIterator1 const first, ForwardIterator2 const d_first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::transform_inclusive_scan_forward_iterator<
            ForwardIterator1, ForwardIterator2, BinaryOperation, UnaryOperation,
            IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
          result_type;
        return result_type(
          first, d_first, binary_operation, unary_operation,
          is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr);
      }


      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation, typename UnaryOperation, typename Value,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      struct transform_inclusive_scan_forward_iterator_init
      {
        ForwardIterator1 first_;
        ForwardIterator2 d_first_;
        BinaryOperation binary_operation_;
        UnaryOperation unary_operation_;
        Value initial_value_;
        IsCalledsPtr is_calleds_ptr_;
        ItersPtr iters_ptr_;
        OutsPtr outs_ptr_;
        PartialSumsPtr partial_sums_ptr_;

        transform_inclusive_scan_forward_iterator_init(
          ForwardIterator1 const first, ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value,
          IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
          PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first),
            binary_operation_(binary_operation),
            unary_operation_(unary_operation),
            initial_value_(initial_value),
            is_calleds_ptr_(is_calleds_ptr),
            iters_ptr_(iters_ptr), outs_ptr_(outs_ptr),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          if (not is_calleds_ptr_[thread_index])
          {
            iters_ptr_[thread_index] = first_;
            outs_ptr_[thread_index] = d_first_;
            std::advance(iters_ptr_[thread_index], n);
            std::advance(outs_ptr_[thread_index], n);

            partial_sums_ptr_[thread_index]
              = thread_index == 0
                ? binary_operation(
                    initial_value_, unary_operation_(*iters_ptr_[thread_index]++))
                : unary_operation_(*iters_ptr_[thread_index]++);
            is_calleds_ptr_[thread_index] = static_cast<int>(true);
          }
          else
            partial_sums_ptr_[thread_index]
              = binary_operation_(
                  partial_sums_ptr_[thread_index],
                  unary_operation_(*iters_ptr_[thread_index]++));

          *outs_ptr_[thread_index]++ = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename ForwardIterator1, typename ForwardIterator2,
        typename BinaryOperation, typename UnaryOperation, typename Value,
        typename IsCalledsPtr, typename ItersPtr, typename OutsPtr,
        typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::transform_inclusive_scan_forward_iterator_init<
        ForwardIterator1, ForwardIterator2, BinaryOperation, UnaryOperation, Value,
        IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
      make_transform_inclusive_scan_forward_iterator_init(
        ForwardIterator1 const first, ForwardIterator2 const d_first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value const initial_value,
        IsCalledsPtr is_calleds_ptr, ItersPtr iters_ptr, OutsPtr outs_ptr,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::transform_inclusive_scan_forward_iterator_init<
            ForwardIterator1, ForwardIterator2, BinaryOperation, UnaryOperation, Value,
            IsCalledsPtr, ItersPtr, OutsPtr, PartialSumsPtr>
          result_type;
        return result_type(
          first, d_first, binary_operation, unary_operation, initial_value,
          is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr);
      }


      template <
        typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename BinaryOperation, typename UnaryOperation, typename PartialSumsPtr>
      struct transform_inclusive_scan_random_access_iterator
      {
        RandomAccessIterator1 first_;
        RandomAccessIterator2 d_first_;
        BinaryOperation binary_operation_;
        UnaryOperation unary_operation_;
        PartialSumsPtr partial_sums_ptr_;

        transform_inclusive_scan_random_access_iterator(
          RandomAccessIterator1 const first, RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          PartialSumsPtr partial_sums_ptr)
          : first_(first), d_first_(d_first),
            binary_operation_(binary_operation),
            unary_operation_(unary_operation),
            partial_sums_ptr_(partial_sums_ptr)
        { }

        typedef void result_type;
        template <typename Size>
        void operator()(Size const n, int const thread_index) const
        {
          partial_sums_ptr_[thread_index]
            = binary_operation_(
                partial_sums_ptr_[thread_index], unary_operation_(first_[n]));
          d_first_[n] = partial_sums_ptr_[thread_index];
        }
      };

      template <
        typename RandomAccessIterator1, typename RandomAccessIterator2,
        typename BinaryOperation, typename UnaryOperation, typename PartialSumsPtr>
      inline ::ket::utility::parallel_loop_n_detail::transform_inclusive_scan_random_access_iterator<
        RandomAccessIterator1, RandomAccessIterator2,
        BinaryOperation, UnaryOperation, PartialSumsPtr>
      make_transform_inclusive_scan_random_access_iterator(
        RandomAccessIterator1 const first, RandomAccessIterator2 const d_first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        PartialSumsPtr partial_sums_ptr)
      {
        typedef
          ::ket::utility::parallel_loop_n_detail::transform_inclusive_scan_random_access_iterator<
            RandomAccessIterator1, RandomAccessIterator2,
            BinaryOperation, UnaryOperation, PartialSumsPtr>
          result_type;
        return result_type(
          first, d_first, binary_operation, unary_operation, partial_sums_ptr);
      }
# endif // BOOST_NO_CXX11_LAMBDAS
    } // namespace parallel_loop_n_detail

    namespace dispatch
    {
      template <typename NumThreads>
      struct transform_inclusive_scan< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation, typename UnaryOperation>
        static ForwardIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        {
          std::vector<int> is_calleds(
            ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          std::vector<ForwardIterator1> iters(
            ::ket::utility::num_threads(parallel_policy));
          std::vector<ForwardIterator2> outs(
            ::ket::utility::num_threads(parallel_policy));
          typedef
            typename std::iterator_traits<ForwardIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy));

          int* is_calleds_ptr = KET_addressof(is_calleds.front());
          ForwardIterator1* iters_ptr = KET_addressof(iters.front());
          ForwardIterator2* outs_ptr = KET_addressof(outs.front());
          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<ForwardIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, unary_operation,
             is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_ptr[thread_index])
              {
                iters_ptr[thread_index] = first;
                outs_ptr[thread_index] = d_first;
                std::advance(iters_ptr[thread_index], n);
                std::advance(outs_ptr[thread_index], n);

                partial_sums_ptr[thread_index]
                  = unary_operation(*iters_ptr[thread_index]++);
                is_calleds_ptr[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_ptr[thread_index]
                  = binary_operation(
                      partial_sums_ptr[thread_index],
                      unary_operation(*iters_ptr[thread_index]++));

              *outs_ptr[thread_index]++ = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::parallel_loop_n_detail::make_transform_inclusive_scan_forward_iterator(
              first, d_first, binary_operation, unary_operation,
              is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds_ptr, partial_sums_ptr, outs_ptr, partial_sums.size());

          return outs.back();
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename BinaryOperation, typename UnaryOperation>
        static RandomAccessIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last,
          RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy), static_cast<value_type>(0));

          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, last-first,
            [first, d_first, binary_operation, unary_operation, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              partial_sums_ptr[thread_index]
                = binary_operation(
                    partial_sums_ptr[thread_index], unary_operation(first[n]));
              d_first[n] = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::parallel_loop_n_detail::make_transform_inclusive_scan_random_access_iterator(
              first, d_first, binary_operation, unary_operation, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last-first, d_first, binary_operation,
            partial_sums_ptr, partial_sums.size());

          return d_first + (last-first);
        }

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static ForwardIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        {
          std::vector<int> is_calleds(
            ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          std::vector<ForwardIterator1> iters(
            ::ket::utility::num_threads(parallel_policy));
          std::vector<ForwardIterator2> outs(
            ::ket::utility::num_threads(parallel_policy));
          typedef
            typename std::iterator_traits<ForwardIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy));

          int* is_calleds_ptr = KET_addressof(is_calleds.front());
          ForwardIterator1* iters_ptr = KET_addressof(iters.front());
          ForwardIterator2* outs_ptr = KET_addressof(outs.front());
          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<ForwardIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, unary_operation, initial_value,
             is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_ptr[thread_index])
              {
                iters_ptr[thread_index] = first;
                outs_ptr[thread_index] = d_first;
                std::advance(iters_ptr[thread_index], n);
                std::advance(outs_ptr[thread_index], n);

                partial_sums_ptr[thread_index]
                  = thread_index == 0
                    ? binary_operation(
                        initial_value, unary_operation(*iters_ptr[thread_index]++))
                    : unary_operation(*iters_ptr[thread_index]++);
                is_calleds_ptr[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_ptr[thread_index]
                  = binary_operation(
                      partial_sums_ptr[thread_index],
                      unary_operation(*iters_ptr[thread_index]++));

              *outs_ptr[thread_index]++ = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, std::distance(first, last),
            ::ket::utility::parallel_loop_n_detail::make_transform_inclusive_scan_forward_iterator_init(
              first, d_first, binary_operation, unary_operation, initial_value,
              is_calleds_ptr, iters_ptr, outs_ptr, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds_ptr, partial_sums_ptr, outs_ptr, partial_sums.size());

          return outs.back();
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static RandomAccessIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last,
          RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::value_type
            value_type;
          std::vector<value_type> partial_sums(
            ::ket::utility::num_threads(parallel_policy), static_cast<value_type>(0));
          partial_sums.front() = initial_value;

          value_type* partial_sums_ptr = KET_addressof(partial_sums.front());

          using ::ket::utility::loop_n;
# ifndef BOOST_NO_CXX11_LAMBDAS
          typedef
            typename std::iterator_traits<RandomAccessIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, last-first,
            [first, d_first, binary_operation, unary_operation, partial_sums_ptr](
              difference_type const n, int const thread_index)
            {
              partial_sums_ptr[thread_index]
                = binary_operation(
                    partial_sums_ptr[thread_index], unary_operation(first[n]));
              d_first[n] = partial_sums_ptr[thread_index];
            });
# else // BOOST_NO_CXX11_LAMBDAS
          loop_n(
            parallel_policy, last-first,
            ::ket::utility::parallel_loop_n_detail::make_transform_inclusive_scan_random_access_iterator(
              first, d_first, binary_operation, unary_operation, partial_sums_ptr));
# endif // BOOST_NO_CXX11_LAMBDAS

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last-first, d_first, binary_operation,
            partial_sums_ptr, partial_sums.size());

          return d_first + (last-first);
        }
      };
    } // namespace dispatch
  } // namespace utility
} // namespace ket


# undef KET_addressof
# undef KET_true_type
# undef KET_false_type
# undef KET_RVALUE_REFERENCE_OR_COPY
# undef KET_FORWARD_OR_COPY

#endif

