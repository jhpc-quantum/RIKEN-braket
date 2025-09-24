#ifndef KET_UTILITY_PARALLEL_LOOP_N_HPP
# define KET_UTILITY_PARALLEL_LOOP_N_HPP

# include <cassert>
# include <vector>
# include <iterator>
# include <algorithm>
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
          : num_threads_{static_cast<NumThreads>(omp_get_max_threads())}
        { }

        explicit parallel(NumThreads const num_threads)
          : num_threads_{
              num_threads <= NumThreads{0}
              ? NumThreads{1}
              : num_threads > static_cast<NumThreads>(omp_get_max_threads())
                ? static_cast<NumThreads>(omp_get_max_threads())
                : num_threads}
        { omp_set_num_threads(static_cast<int>(num_threads_)); }
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
        parallel() noexcept
          : num_threads_{static_cast<NumThreads>(std::thread::hardware_concurrency())}
        { }

        explicit parallel(NumThreads const num_threads) noexcept
          : num_threads_{
              num_threads <= NumThreads{0}
              ? NumThreads{1}
              : num_threads >= static_cast<NumThreads>(std::thread::hardware_concurrency())
                ? static_cast<NumThreads>(std::thread::hardware_concurrency())
                : num_threads}
        { }
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)

        auto num_threads() const noexcept -> NumThreads { return num_threads_; }

# if defined(_OPENMP) && defined(KET_USE_OPENMP)
        auto num_threads(NumThreads const num_threads) const noexcept -> void
        {
          if (num_threads <= 0 or num_threads > static_cast<NumThreads>(omp_get_max_threads()))
            return;

          num_threads_ = num_threads;
          omp_set_num_threads(static_cast<int>(num_threads_));
        }
# else // defined(_OPENMP) && defined(KET_USE_OPENMP)
        auto num_threads(NumThreads const num_threads) const noexcept -> void
        {
          if (num_threads <= 0 or num_threads > static_cast<NumThreads>(std::thread::hardware_concurrency()))
            return;

          num_threads_ = num_threads;
        }
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
      }; // class parallel<NumThreads>

      template <typename NumThreads>
      inline constexpr auto make_parallel() noexcept -> ::ket::utility::policy::parallel<NumThreads>
      { return ::ket::utility::policy::parallel<NumThreads>{}; }

      template <typename NumThreads>
      inline constexpr auto make_parallel(NumThreads const num_threads) noexcept
      -> ::ket::utility::policy::parallel<NumThreads>
      { return ::ket::utility::policy::parallel<NumThreads>{num_threads}; }

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
        static auto call(::ket::utility::policy::parallel<NumThreads> const policy) noexcept -> unsigned int
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

        auto lock() noexcept -> void { omp_set_lock(&omp_lock_); }
        auto try_lock() noexcept -> bool { return static_cast<bool>(omp_test_lock(&omp_lock_)); }
        auto unlock() noexcept -> void { omp_unset_lock(&omp_lock_); }
      }; // class omp_mutex

      class omp_nonstandard_exception
        : public std::runtime_error
      {
       public:
        omp_nonstandard_exception()
          : std::runtime_error{"nonstandard exception is thrown in OpenMP block"}
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
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, Function&& function)
        -> void
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
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, Function&& function)
        -> void
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
            function(count, static_cast<int>(num_futures));

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
        auto invoke(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Function&& function)
        -> void
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
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, int const thread_index, Function&& function)
        -> void
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
        static auto call(::ket::utility::policy::parallel<NumThreads> const, Executor&) -> void
        {
#   pragma omp barrier
        }
      }; // struct barrier< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct single_execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Executor, typename Function>
        static auto call(::ket::utility::policy::parallel<NumThreads> const, Executor&, Function&& function) -> void
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

        using parallel_policy_type = ::ket::utility::policy::parallel<NumThreads>;
        friend class loop_n_in_execute<parallel_policy_type>;
        friend class barrier<parallel_policy_type>;
        friend class single_execute<parallel_policy_type>;

       public:
        execute()
          : mutex_{}, cond_{}, barrier_counters_{}
        { barrier_counters_.reserve(64); }

        template <typename Function>
        auto invoke(::ket::utility::policy::parallel<NumThreads> const parallel_policy, Function&& function) -> void
        {
          assert(::ket::utility::num_threads(parallel_policy) > 0u);

          auto const num_threads = static_cast<NumThreads>(::ket::utility::num_threads(parallel_policy));
          barrier_counters_.push_back(num_threads);

          auto const num_futures = num_threads - NumThreads{1u};
          auto futures = std::vector<std::future<void>>{};
          futures.reserve(num_futures);

          for (auto thread_index = NumThreads{0u}; thread_index < num_futures; ++thread_index)
            futures.push_back(std::async(
              std::launch::async,
              [&function, thread_index, this]
              { function(static_cast<int>(thread_index), *this); }));

          function(static_cast<int>(num_futures), *this);

          for (auto const& future: futures)
            future.wait();
        }
      }; // class execute< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct loop_n_in_execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Integer, typename Function>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          Integer const n, int const thread_index, Function&& function)
        -> void
        {
          auto const num_threads = static_cast<NumThreads>(::ket::utility::num_threads(parallel_policy));
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
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ::ket::utility::dispatch::execute< ::ket::utility::policy::parallel<NumThreads> >& executor)
        -> void
        {
          auto const index = static_cast<std::size_t>(executor.barrier_counters_.size()) - std::size_t{1u};

          if (executor.barrier_counters_[index] == 1)
          {
            std::lock_guard<std::mutex> lock{executor.mutex_};
            executor.barrier_counters_.push_back(static_cast<int>(::ket::utility::num_threads(parallel_policy)));
            --executor.barrier_counters_[index];
            executor.cond_.notify_all();
            return;
          }

          auto lock = std::unique_lock<std::mutex>{executor.mutex_};
          --executor.barrier_counters_[index];
          executor.cond_.wait(lock, [&executor, index] { return executor.barrier_counters_[index] == 0; });
        }
      }; // struct barrier< ::ket::utility::policy::parallel<NumThreads> >

      template <typename NumThreads>
      struct single_execute< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename Function>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ::ket::utility::dispatch::execute< ::ket::utility::policy::parallel<NumThreads> >& executor,
          Function&& function)
        -> void
        {
          auto const index = static_cast<std::size_t>(executor.barrier_counters_.size()) - std::size_t{1u};

          if (executor.barrier_counters_[index]
              == static_cast<int>(::ket::utility::num_threads(parallel_policy)))
          {
            std::lock_guard<std::mutex> lock{executor.mutex_};
            function();
          }

          if (executor.barrier_counters_[index] == 1)
          {
            std::lock_guard<std::mutex> lock{executor.mutex_};
            executor.barrier_counters_.push_back(static_cast<int>(::ket::utility::num_threads(parallel_policy)));
            --executor.barrier_counters_[index];
            executor.cond_.notify_all();
            return;
          }

          auto lock = std::unique_lock<std::mutex>{executor.mutex_};
          --executor.barrier_counters_[index];
          executor.cond_.wait(lock, [&executor, index] { return executor.barrier_counters_[index] == 0; });
        }
      }; // struct single_execute< ::ket::utility::policy::parallel<NumThreads> >
# endif // defined(_OPENMP) && defined(KET_USE_OPENMP)
    } // namespace dispatch


    // copy
    namespace dispatch
    {
      template <typename NumThreads>
      struct copy< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator1, typename ForwardIterator2>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads, first);
          auto d_iters = std::vector<ForwardIterator2>(num_threads, d_first);

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [&is_calleds, &iters, &d_iters](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);
                std::advance(d_iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              *d_iters[thread_index]++ = *iters[thread_index]++;
            });

          return d_iters.back();
        }

        template <typename RandomAccessIterator, typename ForwardIterator>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, ForwardIterator const d_first,
          std::random_access_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto d_iters = std::vector<ForwardIterator>(num_threads, d_first);

          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, &is_calleds, &d_iters](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(d_iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              *d_iters[thread_index]++ = first[n];
            });

          return d_iters.back();
        }

        template <typename ForwardIterator, typename RandomAccessIterator>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, RandomAccessIterator const d_first,
          std::forward_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads, first);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          auto const range_size = std::distance(first, last);
          ::ket::utility::loop_n(
            parallel_policy, range_size,
            [d_first, &is_calleds, &iters](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              d_first[n] = *iters[thread_index]++;
            });

          return d_first + range_size;
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last, RandomAccessIterator2 const d_first,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          auto const range_size = last - first;
          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, range_size,
            [first, d_first](difference_type const n, int) { d_first[n] = first[n]; });

          return d_first + range_size;
        }
      }; // struct copy< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    // copy_if
    namespace dispatch
    {
      template <typename NumThreads>
      struct copy_if< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator1, typename ForwardIterator2, typename UnaryPredicate>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          UnaryPredicate const unary_predicate,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          auto d_steps = std::vector<difference_type>(num_threads, difference_type{0});

          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads, first);

          auto const range_size = std::distance(first, last);
          ::ket::utility::loop_n(
            parallel_policy, range_size,
            [unary_predicate, &d_steps, &is_calleds, &iters](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              if (unary_predicate(*iters[thread_index]))
                ++d_steps[thread_index];
            });

          auto d_iters = std::vector<ForwardIterator2>(num_threads, d_first);
          using std::begin;
          using std::end;
          std::transform(
            begin(d_iters), std::prev(end(d_iters)), begin(d_steps), std::next(begin(d_iters)),
            [](ForwardIterator2 d_iter, difference_type const d_step)
            { std::advance(d_iter, d_step); return d_iter; });

          std::fill(begin(is_calleds), end(is_calleds), static_cast<int>(false));
          std::fill(begin(iters), end(iters), first);

          ::ket::utility::loop_n(
            parallel_policy, range_size,
            [unary_predicate, &is_calleds, &iters, &d_iters](difference_type const n, int thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              if (unary_predicate(*iters[thread_index]))
                *d_iters[thread_index]++ = *iters[thread_index]++;
            });

          return d_iters.back();
        }

        template <typename RandomAccessIterator, typename ForwardIterator, typename UnaryPredicate>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, ForwardIterator const d_first,
          UnaryPredicate const unary_predicate,
          std::random_access_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          auto d_steps = std::vector<difference_type>(num_threads, difference_type{0});

          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, unary_predicate, &d_steps](difference_type const n, int const thread_index)
            {
              if (unary_predicate(first[n]))
                ++d_steps[thread_index];
            });

          auto d_iters = std::vector<ForwardIterator>(num_threads, d_first);
          using std::begin;
          using std::end;
          std::transform(
            begin(d_iters), std::prev(end(d_iters)), begin(d_steps), std::next(begin(d_iters)),
            [](ForwardIterator d_iter, difference_type const d_step)
            { std::advance(d_iter, d_step); return d_iter; });

          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, unary_predicate, &d_iters](difference_type const n, int thread_index)
            {
              if (unary_predicate(first[n]))
                *d_iters[thread_index]++ = first[n];
            });

          return d_iters.back();
        }

        template <typename ForwardIterator, typename RandomAccessIterator, typename UnaryPredicate>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, RandomAccessIterator const d_first,
          UnaryPredicate const unary_predicate,
          std::forward_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          auto d_first_indices = std::vector<difference_type>(num_threads, difference_type{0});

          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads, first);

          auto const range_size = std::distance(first, last);
          ::ket::utility::loop_n(
            parallel_policy, range_size,
            [unary_predicate, &d_first_indices, &is_calleds, &iters](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              if (unary_predicate(*iters[thread_index]))
                ++d_first_indices[thread_index];
            });

          using std::begin;
          using std::end;
          std::partial_sum(begin(d_first_indices), end(d_first_indices), begin(d_first_indices));

          auto d_iters = std::vector<RandomAccessIterator>(num_threads, d_first);
          std::transform(std::next(begin(d_iters)), end(d_iters), begin(d_first_indices), std::next(begin(d_iters)), std::plus<void>{});

          std::fill(begin(is_calleds), end(is_calleds), static_cast<int>(false));
          std::fill(begin(iters), end(iters), first);

          ::ket::utility::loop_n(
            parallel_policy, range_size,
            [unary_predicate, &is_calleds, &iters, &d_iters](difference_type const n, int thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              if (unary_predicate(*iters[thread_index]))
                *d_iters[thread_index]++ = *iters[thread_index]++;
            });

          return d_iters.back();
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename UnaryPredicate>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last, RandomAccessIterator2 const d_first,
          UnaryPredicate const unary_predicate,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          auto d_first_indices = std::vector<difference_type>(num_threads, difference_type{0});

          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, unary_predicate, &d_first_indices](difference_type const n, int const thread_index)
            {
              if (unary_predicate(first[n]))
                ++d_first_indices[thread_index];
            });

          using std::begin;
          using std::end;
          std::partial_sum(begin(d_first_indices), end(d_first_indices), begin(d_first_indices));

          auto d_iters = std::vector<RandomAccessIterator2>(num_threads, d_first);
          std::transform(std::next(begin(d_iters)), end(d_iters), begin(d_first_indices), std::next(begin(d_iters)), std::plus<void>{});

          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, unary_predicate, &d_iters](difference_type const n, int const thread_index)
            {
              if (unary_predicate(first[n]))
                *d_iters[thread_index]++ = first[n];
            });

          return d_iters.back();
        }
      }; // struct copy_if< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    // copy_n
    namespace dispatch
    {
      template <typename NumThreads>
      struct copy_n< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator1, typename Size, typename ForwardIterator2>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, Size const count, ForwardIterator2 const d_first,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          if (count <= Size{0})
            return d_first;

          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads, first);
          auto d_iters = std::vector<ForwardIterator2>(num_threads, d_first);
          ::ket::utility::loop_n(
            parallel_policy, count,
            [&is_calleds, &iters, &d_iters](Size const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);
                std::advance(d_iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              *d_iters[thread_index]++ = *iters[thread_index]++;
            });

          return d_iters.back();
        }

        template <typename RandomAccessIterator, typename Size, typename ForwardIterator>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, Size const count, ForwardIterator const d_first,
          std::random_access_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator
        {
          if (count <= Size{0})
            return d_first;

          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto d_iters = std::vector<ForwardIterator>(num_threads, d_first);
          ::ket::utility::loop_n(
            parallel_policy, count,
            [first, &is_calleds, &d_iters](Size const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(d_iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              *d_iters[thread_index]++ = first[n];
            });

          return d_iters.back();
        }

        template <typename ForwardIterator, typename Size, typename RandomAccessIterator>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, Size const count, RandomAccessIterator const d_first,
          std::forward_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator
        {
          if (count <= Size{0})
            return d_first;

          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads, first);
          ::ket::utility::loop_n(
            parallel_policy, count,
            [d_first, &is_calleds, &iters](Size const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              d_first[n] = *iters[thread_index]++;
            });

          return d_first + count;
        }

        template <typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, Size const count, RandomAccessIterator2 const d_first,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          if (count <= Size{0})
            return d_first;

          ::ket::utility::loop_n(
            parallel_policy, count,
            [first, d_first](Size const n, int) { d_first[n] = first[n]; });

          return d_first + count;
        }
      }; // struct copy_n< ::ket::utility::policy::parallel<NumThreads> >
    } // namespace dispatch


    // fill
    namespace dispatch
    {
      template <typename NumThreads>
      struct fill< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const& value,
          std::forward_iterator_tag const)
        -> void
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, &value, &is_calleds, &iters](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              *iters[thread_index]++ = value;
            });
        }

        template <typename RandomAccessIterator, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, Value const& value,
          std::random_access_iterator_tag const)
        -> void
        {
          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
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
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          std::forward_iterator_tag const)
        -> typename std::iterator_traits<ForwardIterator>::value_type
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads);
          using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
          auto partial_sums = std::vector<value_type>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, &is_calleds, &iters, &partial_sums](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] += *iters[thread_index]++;
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), value_type{});
        }

        template <typename RandomAccessIterator>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          std::random_access_iterator_tag const)
        -> typename std::iterator_traits<RandomAccessIterator>::value_type
        {
          using value_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
          auto partial_sums = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, &partial_sums](difference_type const n, int const thread_index)
            { partial_sums[thread_index] += first[n]; });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), value_type{});
        }

        template <typename ForwardIterator, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const initial_value,
          std::forward_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, &is_calleds, &iters, &partial_sums](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] += *iters[thread_index]++;
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value);
        }

        template <typename RandomAccessIterator, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, Value const initial_value,
          std::random_access_iterator_tag const)
        -> Value
        {
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, &partial_sums](difference_type const n, int const thread_index)
            { partial_sums[thread_index] += first[n]; });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value);
        }

        template <typename ForwardIterator, typename Value, typename BinaryOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          Value const initial_value, BinaryOperation binary_operation,
          std::forward_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, initial_value, binary_operation, &is_calleds, &iters, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], *iters[thread_index]++);
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_operation);
        }

        template <typename RandomAccessIterator, typename Value, typename BinaryOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last,
          Value const initial_value, BinaryOperation binary_operation,
          std::random_access_iterator_tag const)
        -> Value
        {
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, binary_operation, &partial_sums](difference_type const n, int const thread_index)
            { partial_sums[thread_index] = binary_operation(partial_sums[thread_index], first[n]); });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_operation);
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
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first1, ForwardIterator1 const last1,
          ForwardIterator2 const first2, Value const initial_value,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters1 = std::vector<ForwardIterator1>(num_threads);
          auto iters2 = std::vector<ForwardIterator2>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, &is_calleds, &iters1, &iters2, &partial_sums](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters1[thread_index] = first1;
                std::advance(iters1[thread_index], n);

                iters2[thread_index] = first2;
                std::advance(iters2[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] += *iters1[thread_index]++ * *iters2[thread_index]++;
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value);
        }

        template <typename RandomAccessIterator, typename ForwardIterator, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first1, RandomAccessIterator const last1,
          ForwardIterator const first2, Value const initial_value,
          std::random_access_iterator_tag const, std::forward_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters2 = std::vector<ForwardIterator>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, &is_calleds, &iters2, &partial_sums](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters2[thread_index] = first2;
                std::advance(iters2[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] += first1[n] * *iters2[thread_index]++;
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value);
        }

        template <typename ForwardIterator, typename RandomAccessIterator, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first1, ForwardIterator const last1,
          RandomAccessIterator const first2, Value const initial_value,
          std::forward_iterator_tag const, std::random_access_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters1 = std::vector<ForwardIterator>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, &is_calleds, &iters1, &partial_sums](difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters1[thread_index] = first1;
                std::advance(iters1[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] += *iters1[thread_index]++ * first2[n];
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value);
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first1, RandomAccessIterator1 const last1,
          RandomAccessIterator2 const first2, Value const initial_value,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> Value
        {
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, &partial_sums](difference_type const n, int const thread_index)
            { partial_sums[thread_index] += first1[n] * first2[n]; });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value);
        }

        template <
          typename ForwardIterator1, typename ForwardIterator2, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first1, ForwardIterator1 const last1,
          ForwardIterator2 const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters1 = std::vector<ForwardIterator1>(num_threads);
          auto iters2 = std::vector<ForwardIterator2>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, binary_reduction_operation, binary_transform_operation,
             &is_calleds, &iters1, &iters2, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters1[thread_index] = first1;
                std::advance(iters1[thread_index], n);

                iters2[thread_index] = first2;
                std::advance(iters2[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index]
                = binary_reduction_operation(
                    partial_sums[thread_index],
                    binary_transform_operation(*iters1[thread_index]++, *iters2[thread_index]++));
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename RandomAccessIterator, typename ForwardIterator, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first1, RandomAccessIterator const last1,
          ForwardIterator const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::random_access_iterator_tag const, std::forward_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters2 = std::vector<ForwardIterator>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, binary_reduction_operation, binary_transform_operation, &is_calleds, &iters2, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters2[thread_index] = first2;
                std::advance(iters2[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index]
                = binary_reduction_operation(
                    partial_sums[thread_index], binary_transform_operation(first1[n], *iters2[thread_index]++));
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename ForwardIterator, typename RandomAccessIterator, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first1, ForwardIterator const last1,
          RandomAccessIterator const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::forward_iterator_tag const, std::random_access_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters1 = std::vector<ForwardIterator>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first1, last1),
            [first1, first2, binary_reduction_operation, binary_transform_operation,
             &is_calleds, &iters1, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters1[thread_index] = first1;
                std::advance(iters1[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index]
                = binary_reduction_operation(
                    partial_sums[thread_index], binary_transform_operation(*iters1[thread_index]++, first2[n]));
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first1, RandomAccessIterator1 const last1,
          RandomAccessIterator2 const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> Value
        {
          auto partial_sums = std::vector<Value>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last1 - first1,
            [first1, first2, binary_reduction_operation, binary_transform_operation, &partial_sums](
              difference_type const n, int const thread_index)
            { partial_sums[thread_index] = binary_reduction_operation(partial_sums[thread_index], binary_transform_operation(first1[n], first2[n])); });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <typename ForwardIterator, typename Value, typename BinaryReductionOperation, typename UnaryTransformOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          UnaryTransformOperation unary_transform_operation,
          std::forward_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator>(num_threads);
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, binary_reduction_operation, unary_transform_operation, &is_calleds, &iters, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                std::advance(iters[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index]
                = binary_reduction_operation(partial_sums[thread_index], unary_transform_operation(*iters[thread_index]++));
            });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_reduction_operation);
        }

        template <typename RandomAccessIterator, typename Value, typename BinaryReductionOperation, typename UnaryTransformOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator const first, RandomAccessIterator const last, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          UnaryTransformOperation unary_transform_operation,
          std::random_access_iterator_tag const)
        -> Value
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto partial_sums = std::vector<Value>(num_threads);

          using difference_type = typename std::iterator_traits<RandomAccessIterator>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, binary_reduction_operation, unary_transform_operation, &is_calleds, &partial_sums](
              difference_type const n, int const thread_index)
            { partial_sums[thread_index] = binary_reduction_operation(partial_sums[thread_index], unary_transform_operation(first[n])); });

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), initial_value, binary_reduction_operation);
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
      inline auto post_inclusive_scan(
        ParallelPolicy const parallel_policy,
        RangeSize const range_size, ForwardIterator d_first, BinaryOperation binary_operation,
        std::vector<int, Allocator1>& is_calleds,
        std::vector<Value, Allocator2>& partial_sums,
        std::vector<ForwardIterator, Allocator3>& outs)
      -> void
      {
        using std::begin;
        using std::end;
        std::fill(begin(is_calleds), end(is_calleds), static_cast<int>(false));

        std::partial_sum(begin(partial_sums), end(partial_sums), begin(partial_sums), binary_operation);

        ::ket::utility::loop_n(
          parallel_policy, range_size,
          [d_first, binary_operation, &is_calleds, &partial_sums, &outs](
            RangeSize const n, int const thread_index)
          {
            if (thread_index == 0)
              return;

            if (not static_cast<bool>(is_calleds[thread_index]))
            {
              outs[thread_index] = d_first;
              std::advance(outs[thread_index], n);

              is_calleds[thread_index] = static_cast<int>(true);
            }

            *outs[thread_index]++ = binary_operation(partial_sums[thread_index - 1], *outs[thread_index]);
          });
      }

      template <
        typename ParallelPolicy,
        typename RangeSize, typename RandomAccessIterator, typename BinaryOperation,
        typename Value, typename Allocator>
      inline auto post_inclusive_scan(
        ParallelPolicy const parallel_policy,
        RangeSize const range_size, RandomAccessIterator d_first, BinaryOperation binary_operation,
        std::vector<Value, Allocator>& partial_sums)
      -> void
      {
        using std::begin;
        using std::end;
        std::partial_sum(begin(partial_sums), end(partial_sums), begin(partial_sums), binary_operation);

        ::ket::utility::loop_n(
          parallel_policy, range_size,
          [d_first, binary_operation, &partial_sums](RangeSize const n, int const thread_index)
          {
            if (thread_index == 0)
              return;

            d_first[n] = binary_operation(partial_sums[thread_index - 1], d_first[n]);
          });
      }
    } // namespace parallel_loop_n_detail

    namespace dispatch
    {
      template <typename NumThreads>
      struct inclusive_scan< ::ket::utility::policy::parallel<NumThreads> >
      {
        template <typename ForwardIterator1, typename ForwardIterator2>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads);
          auto outs = std::vector<ForwardIterator2>(num_threads);
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, &is_calleds, &iters, &outs, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                outs[thread_index] = d_first;
                std::advance(iters[thread_index], n);
                std::advance(outs[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] += *iters[thread_index]++;
              *outs[thread_index]++ = partial_sums[thread_index];
            });

          using std::begin;
          using std::end;
          std::partial_sum(begin(partial_sums), end(partial_sums), begin(partial_sums));
          std::fill(begin(is_calleds), end(is_calleds), static_cast<int>(false));

          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [d_first, &is_calleds, &outs, &partial_sums](difference_type const n, int const thread_index)
            {
              if (thread_index == 0)
                return;

              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                outs[thread_index] = d_first;
                std::advance(outs[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              *outs[thread_index]++ += partial_sums[thread_index - 1];
            });

          return outs.back();
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last, RandomAccessIterator2 const d_first,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, d_first, &partial_sums](difference_type const n, int const thread_index)
            {
              partial_sums[thread_index] += first[n];
              d_first[n] = partial_sums[thread_index];
            });

          using std::begin;
          using std::end;
          std::partial_sum(begin(partial_sums), end(partial_sums), begin(partial_sums));

          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [d_first, &partial_sums](difference_type const n, int const thread_index)
            {
              if (thread_index == 0)
                return;

              d_first[n] += partial_sums[thread_index - 1];
            });

          return d_first + (last - first);
        }

        template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          BinaryOperation binary_operation, std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads);
          auto outs = std::vector<ForwardIterator2>(num_threads);
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, &is_calleds, &iters, &outs, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                outs[thread_index] = d_first;
                std::advance(iters[thread_index], n);
                std::advance(outs[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], *iters[thread_index]++);
              *outs[thread_index]++ = partial_sums[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation, is_calleds, partial_sums, outs);

          return outs.back();
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename BinaryOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last, RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, &partial_sums](difference_type const n, int const thread_index)
            {
              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], first[n]);
              d_first[n] = partial_sums[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last - first, d_first, binary_operation, partial_sums);

          return d_first + (last - first);
        }

        template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          BinaryOperation binary_operation, Value const initial_value,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads);
          auto outs = std::vector<ForwardIterator2>(num_threads);
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(num_threads);
          partial_sums.front() = binary_operation(initial_value, value_type{});

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, &is_calleds, &iters, &outs, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                outs[thread_index] = d_first;
                std::advance(iters[thread_index], n);
                std::advance(outs[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], *iters[thread_index]++);
              *outs[thread_index]++ = partial_sums[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation, is_calleds, partial_sums, outs);

          return outs.back();
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename BinaryOperation, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last, RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, Value const initial_value,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));
          partial_sums.front() = binary_operation(initial_value, value_type{});

          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, &partial_sums](difference_type const n, int const thread_index)
            {
              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], first[n]);
              d_first[n] = partial_sums[thread_index];
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
        template <typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads);
          auto outs = std::vector<ForwardIterator2>(num_threads);
          using value_type = typename std::iterator_traits<ForwardIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(num_threads);

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, unary_operation, &is_calleds, &iters, &outs, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                outs[thread_index] = d_first;
                std::advance(iters[thread_index], n);
                std::advance(outs[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], unary_operation(*iters[thread_index]++));
              *outs[thread_index]++ = partial_sums[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation, is_calleds, partial_sums, outs);

          return outs.back();
        }

        template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename BinaryOperation, typename UnaryOperation>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last, RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          using value_type = typename std::iterator_traits<RandomAccessIterator1>::value_type;
          auto partial_sums = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));

          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, unary_operation, &partial_sums](difference_type const n, int const thread_index)
            {
              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], unary_operation(first[n]));
              d_first[n] = partial_sums[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, last - first, d_first, binary_operation, partial_sums);

          return d_first + (last - first);
        }

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last, ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation, Value const initial_value,
          std::forward_iterator_tag const, std::forward_iterator_tag const)
        -> ForwardIterator2
        {
          auto const num_threads = ::ket::utility::num_threads(parallel_policy);
          auto is_calleds = std::vector<int>(num_threads, static_cast<int>(false));
          auto iters = std::vector<ForwardIterator1>(num_threads);
          auto outs = std::vector<ForwardIterator2>(num_threads);
          using value_type = typename std::iterator_traits<ForwardIterator2>::value_type;
          auto partial_sums = std::vector<value_type>(num_threads);
          using unary_operation_result_type = std::remove_reference_t<std::remove_cv_t<decltype(unary_operation(*first))>>;
          partial_sums.front() = binary_operation(initial_value, unary_operation_result_type{});

          using difference_type = typename std::iterator_traits<ForwardIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, std::distance(first, last),
            [first, d_first, binary_operation, unary_operation, &is_calleds, &iters, &outs, &partial_sums](
              difference_type const n, int const thread_index)
            {
              if (not static_cast<bool>(is_calleds[thread_index]))
              {
                iters[thread_index] = first;
                outs[thread_index] = d_first;
                std::advance(iters[thread_index], n);
                std::advance(outs[thread_index], n);

                is_calleds[thread_index] = static_cast<int>(true);
              }

              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], unary_operation(*iters[thread_index]++));
              *outs[thread_index]++ = partial_sums[thread_index];
            });

          ::ket::utility::parallel_loop_n_detail::post_inclusive_scan(
            parallel_policy, std::distance(first, last), d_first, binary_operation, is_calleds, partial_sums, outs);

          return outs.back();
        }

        template <
          typename RandomAccessIterator1, typename RandomAccessIterator2,
          typename BinaryOperation, typename UnaryOperation, typename Value>
        static auto call(
          ::ket::utility::policy::parallel<NumThreads> const parallel_policy,
          RandomAccessIterator1 const first, RandomAccessIterator1 const last, RandomAccessIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation, Value const initial_value,
          std::random_access_iterator_tag const, std::random_access_iterator_tag const)
        -> RandomAccessIterator2
        {
          using value_type = typename std::iterator_traits<RandomAccessIterator2>::value_type;
          auto partial_sums = std::vector<value_type>(::ket::utility::num_threads(parallel_policy));
          using unary_operation_result_type = std::remove_reference_t<std::remove_cv_t<decltype(unary_operation(*first))>>;
          partial_sums.front() = binary_operation(initial_value, unary_operation_result_type{});

          using difference_type = typename std::iterator_traits<RandomAccessIterator1>::difference_type;
          ::ket::utility::loop_n(
            parallel_policy, last - first,
            [first, d_first, binary_operation, unary_operation, &partial_sums](difference_type const n, int const thread_index)
            {
              partial_sums[thread_index] = binary_operation(partial_sums[thread_index], unary_operation(first[n]));
              d_first[n] = partial_sums[thread_index];
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
