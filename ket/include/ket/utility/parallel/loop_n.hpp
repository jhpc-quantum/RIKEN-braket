#ifndef KET_UTILITY_PARALLEL_LOOP_N_HPP
# define KET_UTILITY_PARALLEL_LOOP_N_HPP

# include <cassert>
# include <vector>
# include <iterator>
# include <numeric>
# include <utility>
# include <mutex> // lock_guard and unique_lock; mutex unless using OpenMP
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
#   include <stdexcept>
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
#   include <thread>
#   include <future>
#   include <condition_variable>
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
# include <type_traits>

# if defined(_OPENMP) && defined(KET_USE_OPENMP)
#   include <omp.h>

#   include <boost/optional.hpp>
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)

# include <ket/utility/loop_n.hpp>


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
        parallel() noexcept
          : num_threads_(static_cast<NumThreads>(omp_get_max_threads()))
        { }

        explicit parallel(NumThreads const num_threads)
          : num_threads_(
              num_threads <= NumThreads{0}
              ? NumThreads{1}
              : num_threads > static_cast<NumThreads>(omp_get_max_threads())
                ? static_cast<NumThreads>(omp_get_max_threads())
                : num_threads)
        { omp_set_num_threads(static_cast<int>(num_threads_)); }
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
        parallel() noexcept
          : num_threads_(static_cast<NumThreads>(std::thread::hardware_concurrency()))
        { }

        explicit parallel(NumThreads const num_threads) noexcept
          : num_threads_(
              num_threads <= NumThreads{0}
              ? NumThreads{1}
              : num_threads >= static_cast<NumThreads>(std::thread::hardware_concurrency())
                ? static_cast<NumThreads>(std::thread::hardware_concurrency())
                : num_threads)
        { }
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)

        NumThreads num_threads() const noexcept { return num_threads_; }

# if defined(_OPENMP) && defined(KET_USE_OPENMP)
        void num_threads(NumThreads const num_threads) const noexcept
        {
          if (num_threads <= 0 or num_threads > static_cast<NumThreads>(omp_get_max_threads()))
            return;

          num_threads_ = num_threads;
          omp_set_num_threads(static_cast<int>(num_threads_));
        }
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
        void num_threads(NumThreads const num_threads) const noexcept
        {
          if (num_threads <= 0 or num_threads > static_cast<NumThreads>(std::thread::hardware_concurrency()))
            return;

          num_threads_ = num_threads;
        }
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
      }; // class parallel<NumThreads>

      template <typename NumThreads>
      inline constexpr ::ket::utility::policy::parallel<NumThreads> make_parallel() noexcept
      { return ::ket::utility::policy::parallel<NumThreads>{}; }

      template <typename NumThreads>
      inline constexpr ::ket::utility::policy::parallel<NumThreads>
      make_parallel(NumThreads const num_threads) noexcept
      { return ::ket::utility::policy::parallel<NumThreads>(num_threads); }

      namespace meta
      {
        template <typename NumThreads>
        struct is_loop_n_policy< ::ket::utility::policy::parallel<NumThreads> >
          : std::true_type
        { }; // struct is_loop_n_policy< ::ket::utility::policy::parallel<NumThreads> >
      } // namespace meta
    } // namespace policy


    namespace dispatch
    {
      template <typename NumThreads>
      struct num_threads< ::ket::utility::policy::parallel<NumThreads> >
      {
        static unsigned int call(
          ::ket::utility::policy::parallel<NumThreads> const policy) noexcept
        { return static_cast<unsigned int>(policy.num_threads()); }
      }; // struct num_threads< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    namespace parallel_loop_n_detail
    {
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
      class omp_mutex
      {
        omp_lock_t omp_lock_;

       public:
        omp_mutex() noexcept { omp_init_lock(&omp_lock_); }
        ~omp_mutex() noexcept { omp_destroy_lock(&omp_lock_); }

        omp_mutex(omp_mutex const&) = delete;
        omp_mutex& operator=(omp_mutex const&) = delete;
        omp_mutex(omp_mutex&&) = delete;
        omp_mutex& operator=(omp_mutex&&) = delete;

        void lock() noexcept { omp_set_lock(&omp_lock_); }
        bool try_lock() noexcept { return static_cast<bool>(omp_test_lock(&omp_lock_)); }
        void unlock() noexcept { omp_unset_lock(&omp_lock_); }
      }; // class omp_mutex

      class omp_nonstandard_exception
        : public std::runtime_error
      {
       public:
        omp_nonstandard_exception()
          : std::runtime_error("nonstandard exception is thrown in OpenMP block")
        { }
      }; // class omp_nonstandard_exception
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
    } // namespace parallel_loop_n_detail

    namespace dispatch
    {
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
      template <typename NumThreads, typename Integer>
      struct loop_n< ::ket::utility::policy::parallel<NumThreads>, Integer >
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, Function&& function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

          auto maybe_error = boost::optional<std::exception>{};
          auto is_nonstandard_exception_thrown = false;

          using mutex_type = ::ket::utility::parallel_loop_n_detail::omp_mutex;
          mutex_type mutex;

#   pragma omp parallel reduction(||:is_nonstandard_exception_thrown)
          {
            auto const thread_index = static_cast<int>(omp_get_thread_num());
#   pragma omp for
            for (auto count = Integer{0}; count < n; ++count)
              try
              {
                function(count, thread_index);
              }
              catch (std::exception& error)
              {
                std::lock_guard<mutex_type> lock{mutex};

                if (!maybe_error)
                  maybe_error = error;
              }
              catch (...)
              {
                is_nonstandard_exception_thrown = true;
              }
          }

          if (is_nonstandard_exception_thrown)
            throw ::ket::utility::parallel_loop_n_detail::omp_nonstandard_exception{};

          if (maybe_error)
            throw *maybe_error;
        }
      }; // struct loop_n< ::ket::utility::policy::parallel<NumThreads>, Integer >
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
      template <typename NumThreads, typename Integer>
      struct loop_n< ::ket::utility::policy::parallel<NumThreads>, Integer >
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, Function&& function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

          auto const num_threads
            = static_cast<NumThreads>(::ket::utility::num_threads(parallel_policy));
          auto const num_futures = num_threads - NumThreads{1u};
          auto futures = std::vector<std::future<void>>{};
          futures.reserve(num_futures);

          auto const local_num_counts = static_cast<NumThreads>(n) / num_threads;
          auto const remainder = static_cast<NumThreads>(n) % num_threads;
          auto first_count = Integer{0};

          for (auto thread_index = NumThreads{0u}; thread_index < num_futures; ++thread_index)
          {
            auto const last_count
              = static_cast<Integer>(
                  local_num_counts * (thread_index + NumThreads{1u})
                  + std::min(remainder, thread_index + NumThreads{1u}));

            futures.push_back(std::async(
              std::launch::async,
              [&function, first_count, last_count, thread_index]
              {
                for (auto count = first_count; count < last_count; ++count)
                  function(count, static_cast<int>(thread_index));
              }));

            first_count = last_count;
          }

          auto const last_count
            = static_cast<Integer>(
                local_num_counts * num_threads + std::min(remainder, num_threads));

          for (auto count = first_count; count < last_count; ++count)
            std::forward<Function>(function)(count, static_cast<int>(num_futures));

          for (auto const& future: futures)
            future.wait();
        }
      }; // struct loop_n< ::ket::utility::policy::parallel<NumThreads>, Integer >
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
    } // namespace dispatch


    // execute
    namespace dispatch
    {
# if defined(_OPENMP) && defined(KET_USE_OPENMP)
      template <typename NumThreads>
      class execute< ::ket::utility::policy::parallel<NumThreads> >
      {
       public:
        template <typename Function>
        void invoke(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Function&& function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

          auto maybe_error = boost::optional<std::exception>{};
          auto is_nonstandard_exception_thrown = false;

          using mutex_type = ::ket::utility::parallel_loop_n_detail::omp_mutex;
          mutex_type mutex;

#   pragma omp parallel reduction(||:is_nonstandard_exception_thrown)
          {
            try
            {
              function(omp_get_thread_num(), *this);
            }
            catch (std::exception& error)
            {
              std::lock_guard<mutex_type> lock{mutex};

              if (!maybe_error)
                maybe_error = error;
            }
            catch (...)
            {
              is_nonstandard_exception_thrown = true;
            }
          }

          if (is_nonstandard_exception_thrown)
            throw ::ket::utility::parallel_loop_n_detail::omp_nonstandard_exception{};

          if (maybe_error)
            throw *maybe_error;
        }
      }; // class execute< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct loop_n_in_execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Integer, typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, int const thread_index, Function&& function)
        {
#   pragma omp for
          for (auto count = Integer{0}; count < n; ++count)
            function(count, thread_index);
        }
      }; // struct loop_n_in_execute< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct barrier< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Executor>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const, Executor&)
        {
#   pragma omp barrier
        }
      }; // struct barrier< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct single_execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Executor, typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const, Executor&,
          Function&& function)
        {
#   pragma omp single
          {
            function();
          }
        }
      }; // struct single_execute< ::ket::utility::policy::parallel<NumThreads> >
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
      template <typename NumThreads>
      class execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        std::mutex mutex_;
        std::condition_variable cond_;
        std::vector<int> barrier_counters_;

        typedef ::ket::utility::policy::parallel<NumThreads> parallel_policy_type;
        friend class loop_n_in_execute<parallel_policy_type>;
        friend class barrier<parallel_policy_type>;
        friend class single_execute<parallel_policy_type>;

       public:
        execute()
          : mutex_{}, cond_{}, barrier_counters_{}
        { barrier_counters_.reserve(64); }

        template <typename Function>
        void invoke(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Function&& function)
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

          auto const num_threads
            = static_cast<NumThreads>(::ket::utility::num_threads(parallel_policy));
          barrier_counters_.push_back(num_threads);

          auto const num_futures = num_threads - NumThreads{1u};
          auto futures = std::vector<std::future<void>>{};
          futures.reserve(num_futures);

          for (auto thread_index = NumThreads{0u}; thread_index < num_futures; ++thread_index)
          {
            futures.push_back(std::async(
              std::launch::async,
              [&function, thread_index, this]
              { function(static_cast<int>(thread_index), *this); }));
          }

          std::forward<Function>(function)(static_cast<int>(num_futures), *this);

          for (auto const& future: futures)
            future.wait();
        }
      }; // class execute< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct loop_n_in_execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Integer, typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, int const thread_index,
          Function&& function)
        {
          auto const num_threads
            = static_cast<NumThreads>(::ket::utility::num_threads(parallel_policy));
          auto const local_num_counts = static_cast<NumThreads>(n) / num_threads;
          auto const remainder = static_cast<NumThreads>(n) % num_threads;
          auto const first_count
            = static_cast<Integer>(
                local_num_counts * static_cast<NumThreads>(thread_index)
                + std::min(remainder, static_cast<NumThreads>(thread_index)));
          auto const last_count
            = static_cast<Integer>(
                local_num_counts * static_cast<NumThreads>(thread_index + 1)
                + std::min(remainder, static_cast<NumThreads>(thread_index + 1)));

          for (auto count = first_count; count < last_count; ++count)
            function(count, thread_index);
        }
      }; // struct loop_n_in_execute< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct barrier< ::ket::utility::policy::parallel<NumThreads> >
      {
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ::ket::utility::dispatch::execute< ::ket::utility::policy::parallel<NumThreads> >& executor)
        {
          auto const index
            = static_cast<std::size_t>(executor.barrier_counters_.size()) - std::size_t{1u};

          if (executor.barrier_counters_[index] == 1)
          {
            std::lock_guard<std::mutex> lock{executor.mutex_};
            executor.barrier_counters_.push_back(
              static_cast<int>(::ket::utility::num_threads(parallel_policy)));
            --executor.barrier_counters_[index];
            executor.cond_.notify_all();
            return;
          }

          auto lock = std::unique_lock<std::mutex>{executor.mutex_};
          --executor.barrier_counters_[index];
          executor.cond_.wait(
            lock, [&executor, index] { return executor.barrier_counters_[index] == 0; });
        }
      }; // struct barrier< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct single_execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ::ket::utility::dispatch::execute< ::ket::utility::policy::parallel<NumThreads> >& executor,
          Function&& function)
        {
          auto const index
            = static_cast<std::size_t>(executor.barrier_counters_.size()) - std::size_t{1u};

          if (executor.barrier_counters_[index]
              == static_cast<int>(::ket::utility::num_threads(parallel_policy)))
          {
            std::lock_guard<std::mutex> lock{executor.mutex_};
            function();
          }

          if (executor.barrier_counters_[index] == 1)
          {
            std::lock_guard<std::mutex> lock{executor.mutex_};
            executor.barrier_counters_.push_back(
              static_cast<int>(::ket::utility::num_threads(parallel_policy)));
            --executor.barrier_counters_[index];
            executor.cond_.notify_all();
            return;
          }

          auto lock = std::unique_lock<std::mutex>{executor.mutex_};
          --executor.barrier_counters_[index];
          executor.cond_.wait(
            lock, [&executor, index] { return executor.barrier_counters_[index] == 0; });
        }
      }; // struct single_execute< ::ket::utility::policy::parallel<NumThreads> >
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
    } // namespace dispatch


    // fill
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
          auto is_calleds
            = std::vector<int>(
                static_cast<int>(::ket::utility::num_threads(parallel_policy)),
                static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, &value, is_calleds_first, iters_first](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds_first[thread_index]))
              {
                iters_first[thread_index] = first;
                std::advance(iters_first[thread_index], n);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              *iters_first[thread_index]++ = value;
            });
        }

        template <typename RandomAccessIterator, typename Value>
        static void call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, Value const& value,
          std::random_access_iterator_tag const)
        {
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          using ::ket::utility::loop_n;
          loop_n(
            parallel_policy, last - first,
            [first, &value](difference_type const n, int) { first[n] = value; });
        }
      }; // struct fill< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    // reduce
    namespace dispatch
    {
      template <typename NumThreads>
      struct reduce< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator>
        static typename std::iterator_traits<ForwardIterator>::value_type call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, is_calleds_first, iters_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                std::advance(iters_first[thread_index], n);

                partial_sums_first[thread_index] = *iters_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += *iters_first[thread_index]++;
            });

          return std::accumulate(std::next(partial_sums_first), std::end(partial_sums), *partial_sums_first);
        }

        template <typename RandomAccessIterator>
        static typename std::iterator_traits<RandomAccessIterator>::value_type call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          using value_type
            = typename std::iterator_traits<RandomAccessIterator>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, is_calleds_first, partial_sums_first](difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = first[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += first[n];
            });

          return std::accumulate(std::next(partial_sums_first), std::end(partial_sums), *partial_sums_first);
        }

        template <typename ForwardIterator, typename Value>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const initial_value,
          std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, is_calleds_first, iters_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                std::advance(iters_first[thread_index], n);

                partial_sums_first[thread_index] = *iters_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += *iters_first[thread_index]++;
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value);
        }

        template <typename RandomAccessIterator, typename Value>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, Value const initial_value,
          std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = first[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += first[n];
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value);
        }

        template <typename ForwardIterator, typename Value, typename BinaryOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          Value const initial_value, BinaryOperation binary_operation,
          std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, initial_value, binary_operation,
             is_calleds_first, iters_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                std::advance(iters_first[thread_index], n);

                partial_sums_first[thread_index] = *iters_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(
                      partial_sums_first[thread_index], *iters_first[thread_index]++);
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_operation);
        }

        template <typename RandomAccessIterator, typename Value, typename BinaryOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Value const initial_value, BinaryOperation binary_operation,
          std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, binary_operation, is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = first[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(partial_sums_first[thread_index], first[n]);
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_operation);
        }
      }; // struct reduce< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    // transform_reduce
    namespace dispatch
    {
      template <typename NumThreads>
      struct transform_reduce< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator1, typename ForwardIterator2, typename Value>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first1, ForwardIterator1 const last1,
          ForwardIterator2 const first2, Value const initial_value,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters1
            = std::vector<ForwardIterator1>(::ket::utility::num_threads(parallel_policy));
          auto iters2
            = std::vector<ForwardIterator2>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters1_first = std::begin(iters1);
          auto iters2_first = std::begin(iters2);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator1>::difference_type;
          loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, is_calleds_first, iters1_first, iters2_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters1_first[thread_index] = first1;
                std::advance(iters1_first[thread_index], n);

                iters2_first[thread_index] = first2;
                std::advance(iters2_first[thread_index], n);

                partial_sums_first[thread_index] = *iters1_first[thread_index]++ * *iters2_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += *iters1_first[thread_index]++ * *iters2_first[thread_index]++;
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value);
        }

        template <typename RandomAccessIterator, typename ForwardIterator, typename Value>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first1, RandomAccessIterator const last1,
          ForwardIterator const first2, Value const initial_value,
          std::random_access_iterator_tag const, std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters2
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters2_first = std::begin(iters2);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, is_calleds_first, iters2_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters2_first[thread_index] = first2;
                std::advance(iters2_first[thread_index], n);

                partial_sums_first[thread_index] = first1[n] * *iters2_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += first1[n] * *iters2_first[thread_index]++;
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value);
        }

        template <typename ForwardIterator, typename RandomAccessIterator, typename Value>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first1, ForwardIterator const last1,
          RandomAccessIterator const first2, Value const initial_value,
          std::forward_iterator_tag const, std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters1
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters1_first = std::begin(iters1);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator>::difference_type;
          loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, is_calleds_first, iters1_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters1_first[thread_index] = first1;
                std::advance(iters1_first[thread_index], n);

                partial_sums_first[thread_index] = *iters1_first[thread_index]++ * first2[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += *iters1_first[thread_index]++ * first2[n];
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value);
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Value>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first1, RandomAccessIterator1 const last1,
          RandomAccessIterator2 const first2, Value const initial_value,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = first1[n] * first2[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += first1[n] * first2[n];
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value);
        }

        template <
          typename ForwardIterator1, typename ForwardIterator2, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first1, ForwardIterator1 const last1,
          ForwardIterator2 const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters1
            = std::vector<ForwardIterator1>(::ket::utility::num_threads(parallel_policy));
          auto iters2
            = std::vector<ForwardIterator2>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters1_first = std::begin(iters1);
          auto iters2_first = std::begin(iters2);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator1>::difference_type;
          loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, binary_reduction_operation, binary_transform_operation,
             is_calleds_first, iters1_first, iters2_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters1_first[thread_index] = first1;
                std::advance(iters1_first[thread_index], n);

                iters2_first[thread_index] = first2;
                std::advance(iters2_first[thread_index], n);

                partial_sums_first[thread_index]
                  = binary_transform_operation(
                      *iters1_first[thread_index]++, *iters2_first[thread_index]++);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_reduction_operation(
                      partial_sums_first[thread_index],
                      binary_transform_operation(
                        *iters1_first[thread_index]++, *iters2_first[thread_index]++));
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename RandomAccessIterator, typename ForwardIterator, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first1, RandomAccessIterator const last1,
          ForwardIterator const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::random_access_iterator_tag const, std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters2
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters2_first = std::begin(iters2);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, binary_reduction_operation, binary_transform_operation,
             is_calleds_first, iters2_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters2_first[thread_index] = first2;
                std::advance(iters2_first[thread_index], n);

                partial_sums_first[thread_index]
                  = binary_transform_operation(first1[n], *iters2_first[thread_index]++);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_reduction_operation(
                      partial_sums_first[thread_index],
                      binary_transform_operation(first1[n], *iters2_first[thread_index]++));
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename ForwardIterator, typename RandomAccessIterator, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first1, ForwardIterator const last1,
          RandomAccessIterator const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::forward_iterator_tag const, std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters1
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters1_first = std::begin(iters1);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator>::difference_type;
          loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, binary_reduction_operation, binary_transform_operation,
             is_calleds_first, iters1_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters1_first[thread_index] = first1;
                std::advance(iters1_first[thread_index], n);

                partial_sums_first[thread_index]
                  = binary_transform_operation(*iters1_first[thread_index]++, first2[n]);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_reduction_operation(
                      partial_sums_first[thread_index],
                      binary_transform_operation(*iters1_first[thread_index]++, first2[n]));
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first1, RandomAccessIterator1 const last1,
          RandomAccessIterator2 const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, binary_reduction_operation, binary_transform_operation,
             is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = binary_transform_operation(first1[n], first2[n]);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_reduction_operation(
                      partial_sums_first[thread_index],
                      binary_transform_operation(first1[n], first2[n]));
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename ForwardIterator, typename Value,
          typename BinaryReductionOperation, typename UnaryTransformOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          UnaryTransformOperation unary_transform_operation,
          std::forward_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator>(::ket::utility::num_threads(parallel_policy));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, binary_reduction_operation, unary_transform_operation,
             is_calleds_first, iters_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                std::advance(iters_first[thread_index], n);

                partial_sums_first[thread_index]
                  = unary_transform_operation(*iters_first[thread_index]++);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_reduction_operation(
                      partial_sums_first[thread_index],
                      unary_transform_operation(*iters_first[thread_index]++));
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename RandomAccessIterator, typename Value,
          typename BinaryReductionOperation, typename UnaryTransformOperation>
        static Value call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          UnaryTransformOperation unary_transform_operation,
          std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto partial_sums
            = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, binary_reduction_operation, unary_transform_operation,
             is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = unary_transform_operation(first[n]);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_reduction_operation(
                      partial_sums_first[thread_index], unary_transform_operation(first[n]));
            });

          return std::accumulate(partial_sums_first, std::end(partial_sums), initial_value, binary_reduction_operation);
        }
      }; // struct transform_reduce< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    // inclusive_scan
    namespace parallel_loop_n_detail
    {
      template <
        typename ParallelPolicy,
        typename RangeSize, typename ForwardIterator, typename BinaryOperation,
        typename Allocator1, typename Value, typename Allocator2, typename Allocator3>
      inline void post_inclusive_scan(
        ParallelPolicy const parallel_policy,
        RangeSize const range_size, ForwardIterator d_first, BinaryOperation binary_operation,
        std::vector<int, Allocator1>& is_calleds,
        std::vector<Value, Allocator2>& partial_sums,
        std::vector<ForwardIterator, Allocator3>& outs)
      {
        auto is_calleds_first = std::begin(is_calleds);
        auto partial_sums_first = std::begin(partial_sums);
        auto outs_first = std::begin(outs);

        std::fill(is_calleds_first, std::end(is_calleds), static_cast<int>(false));

        std::partial_sum(
          partial_sums_first, std::end(partial_sums),
          partial_sums_first, binary_operation);

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy, range_size,
          [d_first, binary_operation, is_calleds_first, partial_sums_first, outs_first](
            RangeSize const n, int const thread_index)
          {
            if (thread_index == 0)
              return;

            if (not is_calleds_first[thread_index])
            {
              outs_first[thread_index] = d_first;
              std::advance(outs_first[thread_index], n);
              is_calleds_first[thread_index] = static_cast<int>(true);
            }

            *outs_first[thread_index]++
              = binary_operation(
                  partial_sums_first[thread_index - 1], *outs_first[thread_index]);
          });
      }

      template <
        typename ParallelPolicy,
        typename RangeSize, typename RandomAccessIterator, typename BinaryOperation,
        typename Value, typename Allocator>
      inline void post_inclusive_scan(
        ParallelPolicy const parallel_policy,
        RangeSize const range_size, RandomAccessIterator d_first, BinaryOperation binary_operation,
        std::vector<Value, Allocator>& partial_sums)
      {
        auto partial_sums_first = std::begin(partial_sums);

        std::partial_sum(
          partial_sums_first, std::end(partial_sums),
          partial_sums_first, binary_operation);

        using ::ket::utility::loop_n;
        loop_n(
          parallel_policy, range_size,
          [d_first, binary_operation, partial_sums_first](
            RangeSize const n, int const thread_index)
          {
            if (thread_index == 0)
              return;

            d_first[n]
              = binary_operation(partial_sums_first[thread_index - 1], d_first[n]);
          });
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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator1>(::ket::utility::num_threads(parallel_policy));
          auto outs
            = std::vector<ForwardIterator2>(::ket::utility::num_threads(parallel_policy));
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto outs_first = std::begin(outs);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator1>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, is_calleds_first, iters_first, outs_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                outs_first[thread_index] = d_first;
                std::advance(iters_first[thread_index], n);
                std::advance(outs_first[thread_index], n);

                partial_sums_first[thread_index] = *iters_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += *iters_first[thread_index]++;

              *outs_first[thread_index]++ = partial_sums_first[thread_index];
            });

          std::partial_sum(
            partial_sums_first, std::end(partial_sums), partial_sums_first);
          std::fill(
            is_calleds_first, std::end(is_calleds), static_cast<int>(false));

          loop_n(
            parallel_policy, std::distance(first, last),
            [d_first, is_calleds_first, outs_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (thread_index == 0)
                return;

              if (not is_calleds_first[thread_index])
              {
                outs_first[thread_index] = d_first;
                std::advance(outs_first[thread_index], n);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }

              *outs_first[thread_index]++ += partial_sums_first[thread_index - 1];
            });

          return outs.back();
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2>
        static RandomAccessIterator2 call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last,
          RandomAccessIterator2 const d_first,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        {
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          using value_type
            = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, d_first, is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = first[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index] += first[n];

              d_first[n] = partial_sums_first[thread_index];
            });

          std::partial_sum(
            partial_sums_first, std::end(partial_sums), partial_sums_first);

          loop_n(
            parallel_policy, last - first,
            [d_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (thread_index == 0)
                return;

              d_first[n] += partial_sums_first[thread_index - 1];
            });

          return d_first + (last - first);
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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator1>(::ket::utility::num_threads(parallel_policy));
          auto outs
            = std::vector<ForwardIterator2>(::ket::utility::num_threads(parallel_policy));
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto outs_first = std::begin(outs);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator1>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation,
             is_calleds_first, iters_first, outs_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                outs_first[thread_index] = d_first;
                std::advance(iters_first[thread_index], n);
                std::advance(outs_first[thread_index], n);

                partial_sums_first[thread_index] = *iters_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(
                      partial_sums_first[thread_index], *iters_first[thread_index]++);

              *outs_first[thread_index]++ = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds, partial_sums, outs);

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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          using value_type
            = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = first[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(partial_sums_first[thread_index], first[n]);

              d_first[n] = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last - first, d_first, binary_operation, partial_sums);

          return d_first + (last - first);
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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator1>(::ket::utility::num_threads(parallel_policy));
          auto outs
            = std::vector<ForwardIterator2>(::ket::utility::num_threads(parallel_policy));
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto outs_first = std::begin(outs);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator1>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, initial_value,
             is_calleds_first, iters_first, outs_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                outs_first[thread_index] = d_first;
                std::advance(iters_first[thread_index], n);
                std::advance(outs_first[thread_index], n);

                partial_sums_first[thread_index]
                  = thread_index == 0
                    ? binary_operation(initial_value, *iters_first[0]++)
                    : *iters_first[thread_index]++;
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(
                      partial_sums_first[thread_index], *iters_first[thread_index]++);

              *outs_first[thread_index]++ = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds, partial_sums, outs);

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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, initial_value, is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index]
                  = thread_index == 0
                    ? binary_operation(initial_value, first[n])
                    : first[n];
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(partial_sums_first[thread_index], first[n]);

              d_first[n] = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last - first, d_first, binary_operation, partial_sums);

          return d_first + (last - first);
        }
      }; // struct inclusive_scan< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    // transform_inclusive_scan
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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator1>(::ket::utility::num_threads(parallel_policy));
          auto outs
            = std::vector<ForwardIterator2>(::ket::utility::num_threads(parallel_policy));
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto outs_first = std::begin(outs);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<ForwardIterator1>::difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, unary_operation,
             is_calleds_first, iters_first, outs_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                outs_first[thread_index] = d_first;
                std::advance(iters_first[thread_index], n);
                std::advance(outs_first[thread_index], n);

                partial_sums_first[thread_index]
                  = unary_operation(*iters_first[thread_index]++);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(
                      partial_sums_first[thread_index],
                      unary_operation(*iters_first[thread_index]++));

              *outs_first[thread_index]++ = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds, partial_sums, outs);

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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, unary_operation,
             is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index] = unary_operation(first[n]);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(
                      partial_sums_first[thread_index], unary_operation(first[n]));

              d_first[n] = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last - first, d_first, binary_operation, partial_sums);

          return d_first + (last - first);
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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          auto iters
            = std::vector<ForwardIterator1>(::ket::utility::num_threads(parallel_policy));
          auto outs
            = std::vector<ForwardIterator2>(::ket::utility::num_threads(parallel_policy));
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto iters_first = std::begin(iters);
          auto outs_first = std::begin(outs);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          typedef
            typename std::iterator_traits<ForwardIterator1>::difference_type
            difference_type;
          loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, unary_operation, initial_value,
             is_calleds_first, iters_first, outs_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                iters_first[thread_index] = first;
                outs_first[thread_index] = d_first;
                std::advance(iters_first[thread_index], n);
                std::advance(outs_first[thread_index], n);

                partial_sums_first[thread_index]
                  = thread_index == 0
                    ? binary_operation(
                        initial_value, unary_operation(*iters_first[thread_index]++))
                    : unary_operation(*iters_first[thread_index]++);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(
                      partial_sums_first[thread_index],
                      unary_operation(*iters_first[thread_index]++));

              *outs_first[thread_index]++ = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation,
            is_calleds, partial_sums, outs);

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
          auto is_calleds
            = std::vector<int>(
                ::ket::utility::num_threads(parallel_policy), static_cast<int>(false));
          using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums
            = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          auto is_calleds_first = std::begin(is_calleds);
          auto partial_sums_first = std::begin(partial_sums);

          using ::ket::utility::loop_n;
          using difference_type
            = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, unary_operation,
             initial_value, is_calleds_first, partial_sums_first](
              difference_type const n, int const thread_index)
            {
              if (not is_calleds_first[thread_index])
              {
                partial_sums_first[thread_index]
                  = thread_index == 0
                    ? binary_operation(
                        initial_value, unary_operation(first[n]))
                    : unary_operation(first[n]);
                is_calleds_first[thread_index] = static_cast<int>(true);
              }
              else
                partial_sums_first[thread_index]
                  = binary_operation(
                      partial_sums_first[thread_index], unary_operation(first[n]));

              d_first[n] = partial_sums_first[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last - first, d_first, binary_operation, partial_sums);

          return d_first + (last - first);
        }
      }; // struct transform_inclusive_scan< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_PARALLEL_LOOP_N_HPP
