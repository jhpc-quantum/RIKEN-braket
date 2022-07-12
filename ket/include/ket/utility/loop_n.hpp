#ifndef KET_UTILITY_LOOP_N_HPP
# define KET_UTILITY_LOOP_N_HPP

# include <iterator>
# include <algorithm>
# include <numeric>
# include <utility>
# include <type_traits>


namespace ket
{
  namespace utility
  {
    namespace policy
    {
      struct sequential { };

      inline constexpr ::ket::utility::policy::sequential make_sequential() noexcept
      { return ::ket::utility::policy::sequential{}; }

      namespace meta
      {
        template <typename T>
        struct is_loop_n_policy
          : std::false_type
        { }; // struct is_loop_n_policy<T>

        template <>
        struct is_loop_n_policy< ::ket::utility::policy::sequential >
          : std::true_type
        { }; // struct is_loop_n_policy< ::ket::utility::policy::sequential >
      } // namespace meta
    } // namespace policy


    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct num_threads
      {
        static constexpr unsigned int call(ParallelPolicy const);
      }; // struct num_threads<ParallelPolicy>

      template <>
      struct num_threads< ::ket::utility::policy::sequential >
      {
        static constexpr unsigned int call(
          ::ket::utility::policy::sequential const) noexcept
        { return 1u; }
      }; // struct num_threads< ::ket::utility::policy::sequential >
    } // namespace dispatch

    template <typename ParallelPolicy>
    inline constexpr unsigned int num_threads(ParallelPolicy const policy)
      noexcept(
        noexcept(
          ::ket::utility::dispatch::num_threads<ParallelPolicy>::call(policy)))
    { return ::ket::utility::dispatch::num_threads<ParallelPolicy>::call(policy); }


    namespace dispatch
    {
      template <typename ParallelPolicy, typename Integer>
      struct loop_n
      {
        template <typename Function>
        static void call(
          ParallelPolicy const, Integer const n, Function&& function);
      }; // struct loop_n<ParallelPolicy, Integer>

      template <typename Integer>
      struct loop_n< ::ket::utility::policy::sequential, Integer >
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::sequential const,
          Integer const n, Function&& function)
        {
          if (n < Integer{1})
            return;

          for (auto count = Integer{0}; count < n - Integer{1}; ++count)
            function(count, 0);
          std::forward<Function>(function)(n - Integer{1}, 0);
        }
      }; // struct loop_n< ::ket::utility::policy::sequential, Integer >
    } // namespace dispatch

    template <typename Integer, typename Function>
    inline void loop_n(Integer const n, Function&& function)
    {
      using loop_n_type
        = ::ket::utility::dispatch::loop_n< ::ket::utility::policy::sequential, Integer >;
      loop_n_type::call(
        ::ket::utility::policy::sequential(), n, std::forward<Function>(function));
    }

    template <typename ParallelPolicy, typename Integer, typename Function>
    inline void loop_n(
      ParallelPolicy const parallel_policy,
      Integer const n, Function&& function)
    {
      ::ket::utility::dispatch::loop_n<ParallelPolicy, Integer>::call(
        parallel_policy, n, std::forward<Function>(function));
    }


    // execute
    namespace dispatch
    {
      template <typename ParallelPolicy>
      class execute
      {
       public:
        template <typename Function>
        void invoke(ParallelPolicy const, Function&& function);
      }; // class execute<ParallelPolicy>

      template <>
      class execute< ::ket::utility::policy::sequential >
      {
       public:
        template <typename Function>
        void invoke(
          ::ket::utility::policy::sequential const, Function&& function)
        { function(0, *this); }
      }; // class execute< ::ket::utility::policy::sequential >

      template <typename ParallelPolicy>
      struct loop_n_in_execute
      {
        template <typename Integer, typename Function>
        static void call(
          ParallelPolicy const, Integer const, int const, Function&& function);
      }; // struct loop_n_in_execute<ParallelPolicy>

      template <>
      struct loop_n_in_execute< ::ket::utility::policy::sequential >
      {
        template <typename Integer, typename Function>
        static void call(
          ::ket::utility::policy::sequential const,
          Integer const n, int const thread_index, Function&& function)
        {
          if (n < Integer{1})
            return;

          for (auto count = Integer{0}; count < n - Integer{1}; ++count)
            function(count, thread_index);
          std::forward<Function>(function)(n - Integer{1}, thread_index);
        }
      }; // struct loop_n_in_execute< ::ket::utility::policy::sequential >

      template <typename ParallelPolicy>
      struct barrier
      {
        template <typename Executor>
        static void call(ParallelPolicy const, Executor&);
      };

      template <>
      struct barrier< ::ket::utility::policy::sequential >
      {
        static void call(
          ::ket::utility::policy::sequential const,
          ::ket::utility::dispatch::execute< ::ket::utility::policy::sequential >&)
        { }
      }; // struct barrier< ::ket::utility::policy::sequential >

      template <typename ParallelPolicy>
      struct single_execute
      {
        template <typename Executor, typename Function>
        static void call(
          ParallelPolicy const, Executor&, Function&& function);
      }; // struct single_execute<ParallelPolicy>

      template <>
      struct single_execute< ::ket::utility::policy::sequential >
      {
        template <typename Function>
        static void call(
          ::ket::utility::policy::sequential const,
          ::ket::utility::dispatch::execute< ::ket::utility::policy::sequential >&,
          Function&& function)
        { std::forward<Function>(function)(); }
      }; // struct single_execute< ::ket::utility::policy::sequential >
    } // namespace dispatch

    template <typename Function>
    inline void execute(Function&& function)
    {
      using execute_type
        = ::ket::utility::dispatch::execute< ::ket::utility::policy::sequential >;
      execute_type().invoke(
        ::ket::utility::policy::sequential(), std::forward<Function>(function));
    }

    template <typename ParallelPolicy, typename Function>
    inline void execute(
      ParallelPolicy const parallel_policy, Function&& function)
    {
      typedef ::ket::utility::dispatch::execute<ParallelPolicy> execute_type;
      execute_type().invoke(parallel_policy, std::forward<Function>(function));
    }

    template <typename Integer, typename Function>
    inline void loop_n_in_execute(
      Integer const n, int const thread_index, Function&& function)
    {
      using loop_n_in_execute_type
        = ::ket::utility::dispatch ::loop_n_in_execute< ::ket::utility::policy::sequential >;
      loop_n_in_execute_type::call(
        ::ket::utility::policy::sequential(),
        n, thread_index, std::forward<Function>(function));
    }

    template <typename ParallelPolicy, typename Integer, typename Function>
    inline void loop_n_in_execute(
      ParallelPolicy const parallel_policy,
      Integer const n, int const thread_index, Function&& function)
    {
      using loop_n_in_execute_type
        = ::ket::utility::dispatch::loop_n_in_execute<ParallelPolicy>;
      loop_n_in_execute_type::call(
        parallel_policy, n, thread_index, std::forward<Function>(function));
    }

    template <typename Executor>
    inline void barrier(Executor& executor)
    {
      ::ket::utility::dispatch::barrier< ::ket::utility::policy::sequential >::call(
        ::ket::utility::policy::sequential(), executor);
    }

    template <typename ParallelPolicy, typename Executor>
    inline void barrier(ParallelPolicy const parallel_policy, Executor& executor)
    { ::ket::utility::dispatch::barrier<ParallelPolicy>::call(parallel_policy, executor); }

    template <typename Executor, typename Function>
    inline void single_execute(
      Executor& executor, Function&& function)
    {
      ::ket::utility::dispatch::single_execute< ::ket::utility::policy::sequential >::call(
        ::ket::utility::policy::sequential(), executor, std::forward<Function>(function));
    }

    template <typename ParallelPolicy, typename Executor, typename Function>
    inline void single_execute(
      ParallelPolicy const parallel_policy, Executor& executor, Function&& function)
    {
      ::ket::utility::dispatch::single_execute<ParallelPolicy>::call(
        parallel_policy, executor, std::forward<Function>(function));
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
      }; // struct fill<ParallelPolicy>

      template <>
      struct fill< ::ket::utility::policy::sequential >
      {
        template <typename ForwardIterator, typename Value>
        static void call(
          ::ket::utility::policy::sequential const,
          ForwardIterator const first, ForwardIterator const last, Value const& value,
          std::forward_iterator_tag const)
        { return std::fill(first, last, value); }
      }; // struct fill< ::ket::utility::policy::sequential >
    } // namespace dispatch

    template <typename ParallelPolicy, typename ForwardIterator, typename Value>
    inline void fill(
      ParallelPolicy const parallel_policy,
      ForwardIterator const first, ForwardIterator const last, Value const& value)
    {
      ::ket::utility::dispatch::fill<ParallelPolicy>::call(
        parallel_policy, first, last, value,
        typename std::iterator_traits<ForwardIterator>::iterator_category{});
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
        ::ket::utility::fill(parallel_policy, std::begin(range), std::end(range), value);
        return range;
      }

      template <typename ForwardRange, typename Value>
      inline ForwardRange& fill(ForwardRange& range, Value const& value)
      {
        ::ket::utility::fill(std::begin(range), std::end(range), value);
        return range;
      }
    } // namespace ranges


    // reduce
    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct reduce
      {
        template <typename ForwardIterator, typename Category>
        static typename std::iterator_traits<ForwardIterator>::value_type call(
          ParallelPolicy const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          Category const iterator_category);

        template <typename ForwardIterator, typename Value, typename Category>
        static Value call(
          ParallelPolicy const parallel_policy,
          ForwardIterator const first, ForwardIterator const last, Value const initial_value,
          Category const iterator_category);

        template <typename ForwardIterator, typename Value, typename BinaryOperation, typename Category>
        static Value call(
          ParallelPolicy const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          Value const initial_value, BinaryOperation binary_operation,
          Category const iterator_category);
      }; // struct reduce<ParallelPolicy>

      template <>
      struct reduce< ::ket::utility::policy::sequential >
      {
        template <typename InputIterator>
        static typename std::iterator_traits<InputIterator>::value_type call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last,
          std::input_iterator_tag const)
        {
          if (first == last)
            return typename std::iterator_traits<InputIterator>::value_type{};

          return std::accumulate(std::next(first), last, *first);
        }

        template <typename InputIterator, typename Value>
        static Value call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last, Value const initial_value,
          std::input_iterator_tag const)
        { return std::accumulate(first, last, initial_value); }

        template <typename InputIterator, typename Value, typename BinaryOperation>
        static Value call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last,
          Value const initial_value, BinaryOperation binary_operation,
          std::input_iterator_tag const)
        { return std::accumulate(first, last, initial_value, binary_operation); }
      }; // struct reduce< ::ket::utility::policy::sequential >
    } // namespace dispatch

    template <typename ParallelPolicy, typename ForwardIterator>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      typename std::iterator_traits<ForwardIterator>::value_type>::type
    reduce(
      ParallelPolicy const parallel_policy,
      ForwardIterator const first, ForwardIterator const last)
    {
      return ::ket::utility::dispatch::reduce<ParallelPolicy>::call(
        parallel_policy, first, last,
        typename std::iterator_traits<ForwardIterator>::iterator_category{});
    }

    template <typename InputIterator>
    inline typename std::iterator_traits<InputIterator>::value_type
    reduce(InputIterator const first, InputIterator const last)
    {
      if (first == last)
        return typename std::iterator_traits<InputIterator>::value_type{};

      return std::accumulate(std::next(first), last, *first);
    }

    template <typename ParallelPolicy, typename ForwardIterator, typename Value>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, Value>::type
    reduce(
      ParallelPolicy const parallel_policy,
      ForwardIterator const first, ForwardIterator const last, Value const initial_value)
    {
      return ::ket::utility::dispatch::reduce<ParallelPolicy>::call(
        parallel_policy, first, last, initial_value,
        typename std::iterator_traits<ForwardIterator>::iterator_category{});
    }

    template <typename InputIterator, typename Value>
    inline typename std::enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator>::value, Value>::type
    reduce(InputIterator const first, InputIterator const last, Value const initial_value)
    { return std::accumulate(first, last, initial_value); }

    template <typename ParallelPolicy, typename ForwardIterator, typename Value, typename BinaryOperation>
    inline Value reduce(
      ParallelPolicy const parallel_policy,
      ForwardIterator const first, ForwardIterator const last,
      Value const initial_value, BinaryOperation binary_operation)
    {
      return ::ket::utility::dispatch::reduce<ParallelPolicy>::call(
        parallel_policy, first, last, initial_value, binary_operation,
        typename std::iterator_traits<ForwardIterator>::iterator_category{});
    }

    template <typename InputIterator, typename Value, typename BinaryOperation>
    inline typename std::enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator>::value, Value>::type
    reduce(
      InputIterator const first, InputIterator const last,
      Value const initial_value, BinaryOperation binary_operation)
    { return std::accumulate(first, last, initial_value, binary_operation); }

    namespace ranges
    {
      template <typename ParallelPolicy, typename ForwardRange>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
        typename boost::range_value<ForwardRange>::type>::type
      reduce(ParallelPolicy const parallel_policy, ForwardRange const& range)
      { return ::ket::utility::reduce(parallel_policy, std::begin(range), std::end(range)); }

      template <typename ForwardRange>
      inline typename boost::range_value<ForwardRange>::type
      reduce(ForwardRange const& range)
      { return ::ket::utility::reduce(std::begin(range), std::end(range)); }

      template <typename ParallelPolicy, typename ForwardRange, typename Value>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, Value>::type
      reduce(ParallelPolicy const parallel_policy, ForwardRange const& range, Value const initial_value)
      { return ::ket::utility::reduce(parallel_policy, std::begin(range), std::end(range), initial_value); }

      template <typename ForwardRange, typename Value>
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value, Value>::type
      reduce(ForwardRange const& range, Value const initial_value)
      { return ::ket::utility::reduce(std::begin(range), std::end(range), initial_value); }

      template <typename ParallelPolicy, typename ForwardRange, typename Value, typename BinaryOperation>
      inline Value reduce(
        ParallelPolicy const parallel_policy, ForwardRange const& range,
        Value const initial_value, BinaryOperation binary_operation)
      {
        return ::ket::utility::reduce(
          parallel_policy, std::begin(range), std::end(range), initial_value, binary_operation);
      }

      template <typename ForwardRange, typename Value, typename BinaryOperation>
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value, Value>::type
      reduce(ForwardRange const& range, Value const initial_value, BinaryOperation binary_operation)
      { return ::ket::utility::reduce(std::begin(range), std::end(range), initial_value, binary_operation); }
    } // namespace ranges


    // transform_reduce
    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct transform_reduce
      {
        template <typename ForwardIterator1, typename ForwardIterator2, typename Value, typename Category1, typename Category2>
        static Value call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first1, ForwardIterator1 const last1,
          ForwardIterator2 const first2, Value const initial_value,
          Category1 const iterator_category1, Category2 const iterator_category2);

        template <
          typename ForwardIterator1, typename ForwardIterator2, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation,
          typename Category1, typename Category2>
        static Value call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first1, ForwardIterator1 const last1,
          ForwardIterator2 const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          Category1 const iterator_category1, Category2 const iterator_category2);

        template <
          typename ForwardIterator, typename Value,
          typename BinaryReductionOperation, typename UnaryTransformOperation,
          typename Category>
        static Value call(
          ParallelPolicy const parallel_policy,
          ForwardIterator const first, ForwardIterator const last,
          Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          UnaryTransformOperation unary_transform_operation,
          Category const iterator_category);
      }; // struct transform_reduce<ParallelPolicy>

      template <>
      struct transform_reduce< ::ket::utility::policy::sequential >
      {
        template <typename InputIterator1, typename InputIterator2, typename Value>
        static Value call(
          ::ket::utility::policy::sequential const,
          InputIterator1 const first1, InputIterator1 const last1,
          InputIterator2 const first2, Value const initial_value,
          std::input_iterator_tag const, std::input_iterator_tag const)
        { return std::inner_product(first1, last1, first2, initial_value); }

        template <
          typename InputIterator1, typename InputIterator2, typename Value,
          typename BinaryReductionOperation, typename BinaryTransformOperation>
        static Value call(
          ::ket::utility::policy::sequential const,
          InputIterator1 const first1, InputIterator1 const last1,
          InputIterator2 const first2, Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          BinaryTransformOperation binary_transform_operation,
          std::input_iterator_tag const, std::input_iterator_tag const)
        {
          return std::inner_product(
            first1, last1, first2, initial_value,
            binary_reduction_operation, binary_transform_operation);
        }

        template <
          typename InputIterator, typename Value,
          typename BinaryReductionOperation, typename UnaryTransformOperation>
        static Value call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last,
          Value const initial_value,
          BinaryReductionOperation binary_reduction_operation,
          UnaryTransformOperation unary_transform_operation,
          std::input_iterator_tag const)
        {
          using value_type = typename std::iterator_traits<InputIterator>::value_type;
          return std::inner_product(
            first, last, first, initial_value,
            binary_reduction_operation,
            [unary_transform_operation](value_type const& value, value_type const&)
            { return unary_transform_operation(value); });
        }
      }; // struct transform_reduce< ::ket::utility::policy::sequential >
    } // namespace dispatch

    template <typename ParallelPolicy, typename ForwardIterator1, typename ForwardIterator2, typename Value>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, Value>::type
    transform_reduce(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first1, ForwardIterator1 const last1,
      ForwardIterator2 const first2, Value const initial_value)
    {
      return ::ket::utility::dispatch::transform_reduce<ParallelPolicy>::call(
        parallel_policy, first1, last1, first2, initial_value,
        typename std::iterator_traits<ForwardIterator1>::iterator_category{},
        typename std::iterator_traits<ForwardIterator2>::iterator_category{});
    }

    template <typename InputIterator1, typename InputIterator2, typename Value>
    inline Value transform_reduce(
      InputIterator1 const first1, InputIterator1 const last1,
      InputIterator2 const first2, Value const initial_value)
    { return std::inner_product(first1, last1, first2, initial_value); }

    template <
      typename ParallelPolicy, typename ForwardIterator1, typename ForwardIterator2, typename Value,
      typename BinaryReductionOperation, typename BinaryTransformOperation>
    inline Value transform_reduce(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first1, ForwardIterator1 const last1,
      ForwardIterator2 const first2, Value const initial_value,
      BinaryReductionOperation binary_reduction_operation,
      BinaryTransformOperation binary_transform_operation)
    {
      return ::ket::utility::dispatch::transform_reduce<ParallelPolicy>::call(
        parallel_policy, first1, last1, first2, initial_value,
        binary_reduction_operation, binary_transform_operation,
        typename std::iterator_traits<ForwardIterator1>::iterator_category{},
        typename std::iterator_traits<ForwardIterator2>::iterator_category{});
    }

    template <
      typename InputIterator1, typename InputIterator2, typename Value,
      typename BinaryReductionOperation, typename BinaryTransformOperation>
    inline typename std::enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator1>::value, Value>::type
    transform_reduce(
      InputIterator1 const first1, InputIterator1 const last1,
      InputIterator2 const first2, Value const initial_value,
      BinaryReductionOperation binary_reduction_operation,
      BinaryTransformOperation binary_transform_operation)
    {
      return std::inner_product(
        first1, last1, first2, initial_value,
        binary_reduction_operation, binary_transform_operation);
    }

    template <
      typename ParallelPolicy, typename ForwardIterator, typename Value,
      typename BinaryReductionOperation, typename UnaryTransformOperation>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, Value>::type
    transform_reduce(
      ParallelPolicy const parallel_policy,
      ForwardIterator const first, ForwardIterator const last,
      Value const initial_value,
      BinaryReductionOperation binary_reduction_operation,
      UnaryTransformOperation unary_transform_operation)
    {
      return ::ket::utility::dispatch::transform_reduce<ParallelPolicy>::call(
        parallel_policy, first, last, initial_value,
        binary_reduction_operation, unary_transform_operation,
        typename std::iterator_traits<ForwardIterator>::iterator_category{});
    }

    template <
      typename InputIterator, typename Value,
      typename BinaryReductionOperation, typename UnaryTransformOperation>
    inline typename std::enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator>::value, Value>::type
    transform_reduce(
      InputIterator const first, InputIterator const last,
      Value const initial_value,
      BinaryReductionOperation binary_reduction_operation,
      UnaryTransformOperation unary_transform_operation)
    {
      using value_type = typename std::iterator_traits<InputIterator>::value_type;
      return std::inner_product(
        first, last, first, initial_value,
        binary_reduction_operation,
        [unary_transform_operation](value_type const& value, value_type const&)
        { return unary_transform_operation(value); });
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename ForwardRange, typename ForwardIterator, typename Value>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, Value>::type
      transform_reduce(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first, Value const initial_value)
      {
        return ::ket::utility::transform_reduce(
          parallel_policy, std::begin(range), std::end(range), first, initial_value);
      }

      template <typename ForwardRange, typename ForwardIterator, typename Value>
      inline Value transform_reduce(
        ForwardRange const& range, ForwardIterator const first, Value const initial_value)
      { return ::ket::utility::transform_reduce(std::begin(range), std::end(range), first, initial_value); }

      template <
        typename ParallelPolicy, typename ForwardRange, typename ForwardIterator, typename Value,
        typename BinaryReductionOperation, typename BinaryTransformOperation>
      inline Value transform_reduce(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first, Value const initial_value,
        BinaryReductionOperation binary_reduction_operation,
        BinaryTransformOperation binary_transform_operation)
      {
        return ::ket::utility::transform_reduce(
          parallel_policy,
          std::begin(range), std::end(range), first, initial_value,
          binary_reduction_operation, binary_transform_operation);
      }

      template <
        typename ForwardRange, typename ForwardIterator, typename Value,
        typename BinaryReductionOperation, typename BinaryTransformOperation>
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value, Value>::type
      transform_reduce(
        ForwardRange const& range, ForwardIterator const first, Value const initial_value,
        BinaryReductionOperation binary_reduction_operation,
        BinaryTransformOperation binary_transform_operation)
      {
        return ::ket::utility::transform_reduce(
          std::begin(range), std::end(range), first, initial_value,
          binary_reduction_operation, binary_transform_operation);
      }

      template <
        typename ParallelPolicy, typename ForwardRange, typename Value,
        typename BinaryReductionOperation, typename UnaryTransformOperation>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value, Value>::type
      transform_reduce(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, Value const initial_value,
        BinaryReductionOperation binary_reduction_operation,
        UnaryTransformOperation unary_transform_operation)
      {
        return ::ket::utility::transform_reduce(
          parallel_policy,
          std::begin(range), std::end(range), initial_value,
          binary_reduction_operation, unary_transform_operation);
      }

      template <
        typename ForwardRange, typename Value,
        typename BinaryReductionOperation, typename UnaryTransformOperation>
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value, Value>::type
      transform_reduce(
        ForwardRange const& range, Value const initial_value,
        BinaryReductionOperation binary_reduction_operation,
        UnaryTransformOperation unary_transform_operation)
      {
        return ::ket::utility::transform_reduce(
          std::begin(range), std::end(range), initial_value,
          binary_reduction_operation, unary_transform_operation);
      }
    } // namespace ranges


    // inclusive_scan
    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct inclusive_scan
      {
        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename Category1, typename Category2>
        static ForwardIterator2 call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          Category1 const iterator_category1, Category2 const iterator_category2);

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation,
          typename Category1, typename Category2>
        static ForwardIterator2 call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first, BinaryOperation binary_operation,
          Category1 const iterator_category1, Category2 const iterator_category2);

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation, typename Value,
          typename Category1, typename Category2>
        static ForwardIterator2 call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          BinaryOperation binary_operation, Value const initial_value,
          Category1 const iterator_category1, Category2 const iterator_category2);
      }; // struct inclusive_scan<ParallelPolicy>

      template <>
      struct inclusive_scan< ::ket::utility::policy::sequential >
      {
        template <typename InputIterator, typename OutputIterator, typename Category>
        static OutputIterator call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last,
          OutputIterator const d_first,
          std::input_iterator_tag const, Category const)
        { return std::partial_sum(first, last, d_first); }

        template <
          typename InputIterator, typename OutputIterator, typename BinaryOperation,
          typename Category>
        static OutputIterator call(
          ::ket::utility::policy::sequential const,
          InputIterator const first, InputIterator const last,
          OutputIterator const d_first, BinaryOperation binary_operation,
          std::input_iterator_tag const, Category const)
        { return std::partial_sum(first, last, d_first, binary_operation); }

        template <
          typename InputIterator, typename OutputIterator,
          typename BinaryOperation, typename Value, typename Category>
        static OutputIterator call(
          ::ket::utility::policy::sequential const,
          InputIterator first, InputIterator const last, OutputIterator d_first,
          BinaryOperation binary_operation, Value const initial_value,
          std::input_iterator_tag const, Category const)
        {
          if (first == last)
            return d_first;

          auto partial_sum = binary_operation(initial_value, *first);
          *d_first++ = partial_sum;

          while (++first != last)
          {
            partial_sum = binary_operation(partial_sum, *first);
            *d_first++ = partial_sum;
          }

          return d_first;
        }
      }; // struct inclusive_scan< ::ket::utility::policy::sequential >
    } // namespace dispatch

    template <
      typename ParallelPolicy, typename ForwardIterator1, typename ForwardIterator2>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ForwardIterator2>::type
    inclusive_scan(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first, ForwardIterator1 const last,
      ForwardIterator2 const d_first)
    {
      return ::ket::utility::dispatch::inclusive_scan<ParallelPolicy>::call(
        parallel_policy, first, last, d_first,
        typename std::iterator_traits<ForwardIterator1>::iterator_category{},
        typename std::iterator_traits<ForwardIterator2>::iterator_category{});
    }

    template <typename InputIterator, typename OutputIterator>
    inline OutputIterator inclusive_scan(
      InputIterator const first, InputIterator const last,
      OutputIterator const d_first)
    { return std::partial_sum(first, last, d_first); }

    template <
      typename ParallelPolicy,
      typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ForwardIterator2>::type
    inclusive_scan(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first, ForwardIterator1 const last,
      ForwardIterator2 const d_first,
      BinaryOperation binary_operation)
    {
      return ::ket::utility::dispatch::inclusive_scan<ParallelPolicy>::call(
        parallel_policy, first, last, d_first, binary_operation,
        typename std::iterator_traits<ForwardIterator1>::iterator_category{},
        typename std::iterator_traits<ForwardIterator2>::iterator_category{});
    }

    template <typename InputIterator, typename OutputIterator, typename BinaryOperation>
    inline typename std::enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator>::value,
      OutputIterator>::type
    inclusive_scan(
      InputIterator const first, InputIterator const last,
      OutputIterator d_first,
      BinaryOperation binary_operation)
    { return std::partial_sum(first, last, d_first, binary_operation); }

    template <
      typename ParallelPolicy,
      typename ForwardIterator1, typename ForwardIterator2, typename BinaryOperation,
      typename Value>
    inline ForwardIterator2 inclusive_scan(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first, ForwardIterator1 const last,
      ForwardIterator2 const d_first,
      BinaryOperation binary_operation, Value const initial_value)
    {
      return ::ket::utility::dispatch::inclusive_scan<ParallelPolicy>::call(
        parallel_policy, first, last, d_first, binary_operation, initial_value,
        typename std::iterator_traits<ForwardIterator1>::iterator_category{},
        typename std::iterator_traits<ForwardIterator2>::iterator_category{});
    }

    template <
      typename InputIterator, typename OutputIterator, typename BinaryOperation,
      typename Value>
    inline typename std::enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator>::value,
      OutputIterator>::type
    inclusive_scan(
      InputIterator const first, InputIterator const last,
      OutputIterator d_first,
      BinaryOperation binary_operation, Value const initial_value)
    {
      return ::ket::utility::dispatch::inclusive_scan< ::ket::utility::policy::sequential >::call(
        ::ket::utility::policy::make_sequential(),
        first, last, d_first, binary_operation, initial_value,
        typename std::iterator_traits<InputIterator>::iterator_category{},
        typename std::iterator_traits<OutputIterator>::iterator_category{});
    }

    namespace ranges
    {
      template <typename ParallelPolicy, typename ForwardRange, typename ForwardIterator>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
        ForwardIterator>::type
      inclusive_scan(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first)
      {
        return ::ket::utility::inclusive_scan(
          parallel_policy, std::begin(range), std::end(range), first);
      }

      template <typename ForwardRange, typename ForwardIterator>
      inline ForwardIterator inclusive_scan(
        ForwardRange const& range, ForwardIterator const first)
      {
        return ::ket::utility::inclusive_scan(
          std::begin(range), std::end(range), first);
      }

      template <
        typename ParallelPolicy,
        typename ForwardRange, typename ForwardIterator, typename BinaryOperation>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
        ForwardIterator>::type
      inclusive_scan(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation)
      {
        return ::ket::utility::inclusive_scan(
          parallel_policy,
          std::begin(range), std::end(range), first, binary_operation);
      }

      template <
        typename ForwardRange, typename ForwardIterator, typename BinaryOperation>
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value,
        ForwardIterator>::type
      inclusive_scan(
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation)
      {
        return ::ket::utility::inclusive_scan(
          std::begin(range), std::end(range), first, binary_operation);
      }

      template <
        typename ParallelPolicy,
        typename ForwardRange, typename ForwardIterator, typename BinaryOperation,
        typename Value>
      inline ForwardIterator inclusive_scan(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation, Value const initial_value)
      {
        return ::ket::utility::inclusive_scan(
          parallel_policy,
          std::begin(range), std::end(range), first,
          binary_operation, initial_value);
      }

      template <
        typename ForwardRange, typename ForwardIterator, typename BinaryOperation,
        typename Value>
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value,
        ForwardIterator>::type
      inclusive_scan(
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation, Value const initial_value)
      {
        return ::ket::utility::inclusive_scan(
          std::begin(range), std::end(range), first,
          binary_operation, initial_value);
      }
    } // namespace ranges


    // transform inclusive_scan
    namespace dispatch
    {
      template <typename ParallelPolicy>
      struct transform_inclusive_scan
      {
        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation, typename UnaryOperation,
          typename Category1, typename Category2>
        static ForwardIterator2 call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Category1 const iterator_category1, Category2 const iterator_category2);

        template <
          typename ForwardIterator1, typename ForwardIterator2,
          typename BinaryOperation, typename UnaryOperation, typename Value,
          typename Category1, typename Category2>
        static ForwardIterator2 call(
          ParallelPolicy const parallel_policy,
          ForwardIterator1 const first, ForwardIterator1 const last,
          ForwardIterator2 const d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value,
          Category1 const iterator_category1, Category2 const iterator_category2);
      }; // struct transform_inclusive_scan<ParallelPolicy>

      template <>
      struct transform_inclusive_scan< ::ket::utility::policy::sequential >
      {
        template <
          typename InputIterator, typename OutputIterator,
          typename BinaryOperation, typename UnaryOperation, typename Category>
        static OutputIterator call(
          ::ket::utility::policy::sequential const,
          InputIterator first, InputIterator const last, OutputIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          std::input_iterator_tag const, Category const)
        {
          if (first == last)
            return d_first;

          auto partial_sum = unary_operation(*first);
          *d_first++ = partial_sum;

          while (++first != last)
          {
            partial_sum
              = binary_operation(partial_sum, unary_operation(*first));
            *d_first++ = partial_sum;
          }

          return d_first;
        }

        template <
          typename InputIterator, typename OutputIterator,
          typename BinaryOperation, typename UnaryOperation, typename Value,
          typename Category>
        static OutputIterator call(
          ::ket::utility::policy::sequential const,
          InputIterator first, InputIterator const last, OutputIterator d_first,
          BinaryOperation binary_operation, UnaryOperation unary_operation,
          Value const initial_value,
          std::input_iterator_tag const, Category const)
        {
          if (first == last)
            return d_first;

          auto partial_sum = binary_operation(initial_value, unary_operation(*first));
          *d_first++ = partial_sum;

          while (++first != last)
          {
            partial_sum
              = binary_operation(partial_sum, unary_operation(*first));
            *d_first++ = partial_sum;
          }

          return d_first;
        }
      }; // struct transform_inclusive_scan< ::ket::utility::policy::sequential >
    } // namespace dispatch

    template <
      typename ParallelPolicy,
      typename ForwardIterator1, typename ForwardIterator2,
      typename BinaryOperation, typename UnaryOperation>
    inline typename std::enable_if<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ForwardIterator2>::type
    transform_inclusive_scan(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first, ForwardIterator1 const last,
      ForwardIterator2 const d_first,
      BinaryOperation binary_operation, UnaryOperation unary_operation)
    {
      return ::ket::utility::dispatch::transform_inclusive_scan<ParallelPolicy>::call(
        parallel_policy, first, last, d_first, binary_operation, unary_operation,
        typename std::iterator_traits<ForwardIterator1>::iterator_category{},
        typename std::iterator_traits<ForwardIterator2>::iterator_category{});
    }

    template <
      typename InputIterator, typename OutputIterator,
      typename BinaryOperation, typename UnaryOperation>
    inline OutputIterator transform_inclusive_scan(
      InputIterator const first, InputIterator const last,
      OutputIterator d_first,
      BinaryOperation binary_operation, UnaryOperation unary_operation)
    {
      using transform_inclusive_scan_type
        = ::ket::utility::dispatch::transform_inclusive_scan< ::ket::utility::policy::sequential >;
      return transform_inclusive_scan_type::call(
        ::ket::utility::policy::make_sequential(),
        first, last, d_first, binary_operation, unary_operation,
        typename std::iterator_traits<InputIterator>::iterator_category{},
        typename std::iterator_traits<OutputIterator>::iterator_category{});
    }

    template <
      typename ParallelPolicy,
      typename ForwardIterator1, typename ForwardIterator2,
      typename BinaryOperation, typename UnaryOperation, typename Value>
    inline ForwardIterator2 transform_inclusive_scan(
      ParallelPolicy const parallel_policy,
      ForwardIterator1 const first, ForwardIterator1 const last,
      ForwardIterator2 const d_first,
      BinaryOperation binary_operation, UnaryOperation unary_operation,
      Value const initial_value)
    {
      return ::ket::utility::dispatch::transform_inclusive_scan<ParallelPolicy>::call(
        parallel_policy,
        first, last, d_first, binary_operation, unary_operation, initial_value,
        typename std::iterator_traits<ForwardIterator1>::iterator_category{},
        typename std::iterator_traits<ForwardIterator2>::iterator_category{});
    }

    template <
      typename InputIterator, typename OutputIterator,
      typename BinaryOperation, typename UnaryOperation, typename Value>
    inline typename std::enable_if<
      not ::ket::utility::policy::meta::is_loop_n_policy<InputIterator>::value,
      OutputIterator>::type
    transform_inclusive_scan(
      InputIterator const first, InputIterator const last,
      OutputIterator d_first,
      BinaryOperation binary_operation, UnaryOperation unary_operation,
      Value const initial_value)
    {
      using transform_inclusive_scan_type = ::ket::utility::dispatch::transform_inclusive_scan< ::ket::utility::policy::sequential >;
      return transform_inclusive_scan_type::call(
        ::ket::utility::policy::make_sequential(),
        first, last, d_first, binary_operation, unary_operation, initial_value,
        typename std::iterator_traits<InputIterator>::iterator_category{},
        typename std::iterator_traits<OutputIterator>::iterator_category{});
    }

    namespace ranges
    {
      template <
        typename ParallelPolicy,
        typename ForwardRange, typename ForwardIterator,
        typename BinaryOperation, typename UnaryOperation>
      inline typename std::enable_if<
        ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
        ForwardIterator>::type
      transform_inclusive_scan(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation, UnaryOperation unary_operation)
      {
        return ::ket::utility::transform_inclusive_scan(
          parallel_policy,
          std::begin(range), std::end(range), first,
          binary_operation, unary_operation);
      }

      template <
        typename ForwardRange, typename ForwardIterator,
        typename BinaryOperation, typename UnaryOperation>
      inline ForwardIterator transform_inclusive_scan(
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation, UnaryOperation unary_operation)
      {
        return ::ket::utility::transform_inclusive_scan(
          std::begin(range), std::end(range), first,
          binary_operation, unary_operation);
      }

      template <
        typename ParallelPolicy,
        typename ForwardRange, typename ForwardIterator,
        typename BinaryOperation, typename UnaryOperation, typename Value>
      inline ForwardIterator transform_inclusive_scan(
        ParallelPolicy const parallel_policy,
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value const initial_value)
      {
        return ::ket::utility::transform_inclusive_scan(
          parallel_policy,
          std::begin(range), std::end(range), first,
          binary_operation, unary_operation, initial_value);
      }

      template <
        typename ForwardRange, typename ForwardIterator,
        typename BinaryOperation, typename UnaryOperation, typename Value>
      inline typename std::enable_if<
        not ::ket::utility::policy::meta::is_loop_n_policy<ForwardRange>::value,
        ForwardIterator>::type
      transform_inclusive_scan(
        ForwardRange const& range, ForwardIterator const first,
        BinaryOperation binary_operation, UnaryOperation unary_operation,
        Value const initial_value)
      {
        return ::ket::utility::transform_inclusive_scan(
          std::begin(range), std::end(range), first,
          binary_operation, unary_operation, initial_value);
      }
    } // namespace ranges
  } // namespace utility
} // namespace ket


#endif // KET_UTILITY_LOOP_N_HPP
