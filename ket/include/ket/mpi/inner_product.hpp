#ifndef KET_MPI_INNER_PRODUCT_HPP
# define KET_MPI_INNER_PRODUCT_HPP

# include <cassert>
# include <cstddef>
# include <vector>
# include <string>
# include <iterator>
# include <numeric>
# include <type_traits>

# include <boost/optional.hpp>

# include <yampi/environment.hpp>
# include <yampi/datatype_base.hpp>
# include <yampi/communicator.hpp>
# include <yampi/intercommunicator.hpp>
# include <yampi/buffer.hpp>
# include <yampi/send_receive.hpp>
# include <yampi/broadcast.hpp>
# include <yampi/all_reduce.hpp>
# include <yampi/reduce.hpp>
# include <yampi/binary_operation.hpp>

# include <ket/utility/loop_n.hpp>
# include <ket/utility/meta/ranges.hpp>
# include <ket/mpi/page/none_on_page.hpp>
# include <ket/mpi/page/any_on_page.hpp>
# include <ket/mpi/gate/detail/append_qubits_string.hpp>
# include <ket/mpi/utility/simple_mpi.hpp>
# include <ket/mpi/utility/for_each_local_range.hpp>
# include <ket/mpi/utility/resize_buffer_if_empty.hpp>
# include <ket/mpi/utility/logger.hpp>


namespace ket
{
  namespace mpi
  {
    // <Psi_2|Psi_1>
    // (1) two states |Psi_1> and |Psi_2> exist in the same MPI group
    namespace dispatch
    {
      template <typename LocalState1_, typename LocalState2_>
      struct inner_product
      {
        template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2>
        static auto call(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          LocalState1 const& local_state1, LocalState2 const& local_state2,
          yampi::rank const rank_in_unit)
        -> ::ket::utility::meta::range_value_t<LocalState1>
        {
          using complex_type = ::ket::utility::meta::range_value_t<LocalState1>;
          static_assert(
            std::is_same<complex_type, ::ket::utility::meta::range_value_t<LocalState2>>::value,
            "value_type's of LocalState1 and LocalState2 should be the same");
          auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

          auto const data_block_size
            = ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state1, rank_in_unit);
          assert(data_block_size == ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state2, rank_in_unit));
          auto const num_data_blocks
            = ::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit);

          for (auto data_block_index = decltype(num_data_blocks){0u}; data_block_index < num_data_blocks; ++data_block_index)
          {
            using std::begin;
            auto const first1 = begin(local_state1) + data_block_index * data_block_size;
            auto const first2 = begin(local_state2) + data_block_index * data_block_size;

            ::ket::utility::loop_n(
              parallel_policy, data_block_size,
              [&partial_sums, first1, first2](decltype(data_block_size) const index, int const thread_index)
              { using std::conj; partial_sums[thread_index] += conj(*(first2 + index)) * *(first1 + index); });
          }

          using std::begin;
          using std::end;
          return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
        }
      }; // struct inner_product<LocalState1_, LocalState2_>
    } // namespace dispatch

    // all_reduce version
    namespace inner_product_detail
    {
      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1 const& local_state1, LocalState2 const& local_state2,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState1>
      {
        auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);

        using inner_product_impl
          = ::ket::mpi::dispatch::inner_product<std::remove_cv_t<std::remove_reference_t<LocalState1>>, std::remove_cv_t<std::remove_reference_t<LocalState2>>>;
        auto result = inner_product_impl::call(mpi_policy, parallel_policy, local_state1, local_state2, rank_in_unit);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          communicator, environment);

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1 const& local_state1, LocalState2 const& local_state2,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState1>
      {
        auto const rank_in_unit = ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment);

        using inner_product_impl
          = ::ket::mpi::dispatch::inner_product<std::remove_cv_t<std::remove_reference_t<LocalState1>>, std::remove_cv_t<std::remove_reference_t<LocalState2>>>;
        auto result = inner_product_impl::call(mpi_policy, parallel_policy, local_state1, local_state2, rank_in_unit);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          communicator, environment);

        return result;
      }
    } // namespace inner_product_detail

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(mpi_policy, parallel_policy, local_state1, local_state2, communicator, environment);
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(mpi_policy, parallel_policy, local_state1, local_state2, datatype, communicator, environment);
    }

    template <typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, communicator, environment);
    }

    template <typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, datatype, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, datatype, communicator, environment);
    }

    // reduce version
    namespace inner_product_detail
    {
      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1 const& local_state1, LocalState2 const& local_state2,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState1> >
      {
        using inner_product_impl
          = ::ket::mpi::dispatch::inner_product<std::remove_cv_t<std::remove_reference_t<LocalState1>>, std::remove_cv_t<std::remove_reference_t<LocalState2>>>;
        auto result = inner_product_impl::call(mpi_policy, parallel_policy, local_state1, local_state2, communicator, environment);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          root, communicator, environment);

        if (communicator.rank(environment) != root)
          return boost::none;

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1 const& local_state1, LocalState2 const& local_state2,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState1> >
      {
        using inner_product_impl
          = ::ket::mpi::dispatch::inner_product<std::remove_cv_t<std::remove_reference_t<LocalState1>>, std::remove_cv_t<std::remove_reference_t<LocalState2>>>;
        auto result = inner_product_impl::call(mpi_policy, parallel_policy, local_state1, local_state2, communicator, environment);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          root, communicator, environment);

        if (communicator.rank(environment) != root)
          return boost::none;

        return result;
      }
    } // namespace inner_product_detail

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(mpi_policy, parallel_policy, local_state1, local_state2, root, communicator, environment);
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(mpi_policy, parallel_policy, local_state1, local_state2, datatype, root, communicator, environment);
    }

    template <typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, root, communicator, environment);
    }

    template <typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, local_state2, datatype, root, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, root, communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState1, typename LocalState2, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1 const& local_state1, LocalState2 const& local_state2,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, local_state2, datatype, root, communicator, environment);
    }

    // <Psi_{remote}|Psi_{local}>
    // (2) two states |Psi_{local}> and |Psi_{remote}> do not exist in the same MPI group
    // all_reduce version
    namespace inner_product_detail
    {
      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        auto const rank = intracommunicator.rank(environment);
        assert(rank == intercommunicator.rank(environment));
# ifndef NDEBUG
        auto const num_processes = intracommunicator.size(environment);
# endif // NDEBUG
        assert(num_processes == intercommunicator.size(environment) and num_processes == intercommunicator.remote_size(environment));

        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const buffer_last = ::ket::mpi::utility::buffer_end(local_state, buffer);
        auto const buffer_size = buffer_last - buffer_first;

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto result = complex_type{};
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, intracommunicator, environment,
          [parallel_policy, &intercommunicator, &environment,
           rank, buffer_first, buffer_last, buffer_size, &result](
            auto const first, auto const last)
          {
            auto const tag = yampi::tag{rank.mpi_rank()};
            auto iter = first;
            for (; iter + buffer_size < last; iter += buffer_size)
            {
              // <Psi_{remote}|Psi_{local}> = sum_n a_{remote}(n)^* a_{local}(n)
              yampi::send_receive(
                yampi::ignore_status,
                yampi::make_buffer(iter, iter + buffer_size), rank, tag,
                yampi::make_buffer(buffer_first, buffer_last), rank, tag,
                intercommunicator, environment);
              result
                = ::ket::utility::transform_reduce(
                    parallel_policy, iter, iter + buffer_size, buffer_first, result,
                    std::plus<complex_type>{},
                    [](complex_type const& local_coefficient, complex_type const& remote_coefficient)
                    { using std::conj; return conj(remote_coefficient) * local_coefficient; });
            }

            if (iter < last)
            {
              // <Psi_{remote}|Psi_{local}> = sum_n a_{remote}(n)^* a_{local}(n)
              yampi::send_receive(
                yampi::ignore_status,
                yampi::make_buffer(iter, last), rank, tag,
                yampi::make_buffer(buffer_first, buffer_first + (last - iter)), rank, tag,
                intercommunicator, environment);
              result
                = ::ket::utility::transform_reduce(
                    parallel_policy, iter, last, buffer_first, result,
                    std::plus<complex_type>{},
                    [](complex_type const& local_coefficient, complex_type const& remote_coefficient)
                    { using std::conj; return conj(remote_coefficient) * local_coefficient; });
            }
          });

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        auto const rank = intracommunicator.rank(environment);
        assert(rank == intercommunicator.rank(environment));
# ifndef NDEBUG
        auto const num_processes = intracommunicator.size(environment);
# endif // NDEBUG
        assert(num_processes == intercommunicator.size(environment) and num_processes == intercommunicator.remote_size(environment));

        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const buffer_last = ::ket::mpi::utility::buffer_end(local_state, buffer);
        auto const buffer_size = buffer_last - buffer_first;

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto result = complex_type{};
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, intracommunicator, environment,
          [parallel_policy, &datatype, &intercommunicator, &environment,
           rank, buffer_first, buffer_last, buffer_size, &result](
            auto const first, auto const last)
          {
            auto const tag = yampi::tag{rank.mpi_rank()};
            auto const buffer_size = buffer_last - buffer_first;
            auto iter = first;
            for (; iter + buffer_size < last; iter += buffer_size)
            {
              // <Psi_{remote}|Psi_{local}> = sum_n a_{remote}(n)^* a_{local}(n)
              yampi::send_receive(
                yampi::ignore_status,
                yampi::make_buffer(iter, iter + buffer_size, datatype), rank, tag,
                yampi::make_buffer(buffer_first, buffer_last, datatype), rank, tag,
                intercommunicator, environment);
              result
                = ::ket::utility::transform_reduce(
                    parallel_policy, iter, iter + buffer_size, buffer_first, result,
                    std::plus<complex_type>{},
                    [](complex_type const& local_coefficient, complex_type const& remote_coefficient)
                    { using std::conj; return conj(remote_coefficient) * local_coefficient; });
            }

            if (iter < last)
            {
              // <Psi_{remote}|Psi_{local}> = sum_n a_{remote}(n)^* a_{local}(n)
              yampi::send_receive(
                yampi::ignore_status,
                yampi::make_buffer(iter, last, datatype), rank, tag,
                yampi::make_buffer(buffer_first, buffer_first + (last - iter), datatype), rank, tag,
                intercommunicator, environment);
              result
                = ::ket::utility::transform_reduce(
                    parallel_policy, iter, last, buffer_first, result,
                    std::plus<complex_type>{},
                    [](complex_type const& local_coefficient, complex_type const& remote_coefficient)
                    { using std::conj; return conj(remote_coefficient) * local_coefficient; });
            }
          });

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy, local_state, buffer, intracommunicator, intercommunicator, environment);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          intracommunicator, environment);

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename DerivedDatatype, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy, local_state, buffer, datatype, intracommunicator, intercommunicator, environment);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          intracommunicator, environment);

        return result;
      }
    } // namespace inner_product_detail

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, intracommunicator, intercommunicator, environment);
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, datatype, intracommunicator, intercommunicator, environment);
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, intracommunicator, intercommunicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, intracommunicator, intercommunicator, environment);
    }

    // reduce version
    namespace inner_product_detail
    {
      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(intracommunicator.size(environment) == intercommunicator.size(environment));

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy, local_state, buffer, intercommunicator, environment);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          root, intracommunicator, environment);

        if (intracommunicator.rank(environment) != root)
          return boost::none;

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename DerivedDatatype, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(intracommunicator.size(environment) == intercommunicator.size(environment));

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy, local_state, buffer, datatype, intercommunicator, environment);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          root, intracommunicator, environment);

        if (intracommunicator.rank(environment) != root)
          return boost::none;

        return result;
      }
    } // namespace inner_product_detail

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, root, intracommunicator, intercommunicator, environment);
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy, local_state, buffer, datatype, root, intracommunicator, intercommunicator, environment);
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, root, intracommunicator, intercommunicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, root, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, root, intracommunicator, intercommunicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, root, intracommunicator, intercommunicator, environment);
    }

    // <Psi_k|Psi_0> (k = 0, ..., N_{states}: value of intercircuit_communicator.rank(environment))
    // (3) states |Psi_k> do not exist in the same MPI group
    // all_reduce version
    namespace inner_product_detail
    {
      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const buffer_last = ::ket::mpi::utility::buffer_end(local_state, buffer);
        auto const buffer_size = buffer_last - buffer_first;

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto result = complex_type{};
        if (intercircuit_communicator.rank(environment) == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, intercircuit_root, &intercircuit_communicator, &environment, buffer_size, &result](
              auto const first, auto const last)
            {
              auto iter = first;
              for (; iter + buffer_size < last; iter += buffer_size)
              {
                // <Psi_0|Psi_0> = sum_n a_0(n)^* a_0(n) = sum_n |a_0(n)|^2 = 1
                yampi::broadcast(
                  yampi::make_buffer(iter, iter + buffer_size),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, iter + buffer_size, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient0)
                      { using std::norm; return static_cast<complex_type>(norm(coefficient0)); });
              }

              if (iter < last)
              {
                // <Psi_0|Psi_0> = sum_n a_0(n)^* a_0(n) = sum_n |a_0(n)|^2 = 1
                yampi::broadcast(
                  yampi::make_buffer(iter, last),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, last, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient0)
                      { using std::norm; return static_cast<complex_type>(norm(coefficient0)); });
              }
            });
        else // if (intercircuit_communicator.rank(environment) == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, intercircuit_root, &intercircuit_communicator, &environment,
             buffer_first, buffer_last, buffer_size, &result](
              auto const first, auto const last)
            {
              auto iter = first;
              for (; iter + buffer_size < last; iter += buffer_size)
              {
                // <Psi_k|Psi_0> = sum_n a_k(n)^* a_0(n)
                yampi::broadcast(
                  yampi::make_buffer(buffer_first, buffer_last),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, iter + buffer_size, buffer_first, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient_k, complex_type const& coefficient0)
                      { using std::conj; return conj(coefficient_k) * coefficient0; });
              }

              if (iter < last)
              {
                // <Psi_k|Psi_0> = sum_n a_k(n)^* a_0(n)
                yampi::broadcast(
                  yampi::make_buffer(buffer_first, buffer_first + (last - iter)),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, last, buffer_first, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient_k, complex_type const& coefficient0)
                      { using std::conj; return conj(coefficient_k) * coefficient0; });
              }
            });

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const buffer_last = ::ket::mpi::utility::buffer_end(local_state, buffer);
        auto const buffer_size = buffer_last - buffer_first;

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto result = complex_type{};
        if (intercircuit_communicator.rank(environment) == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, &datatype, intercircuit_root, &intercircuit_communicator, &environment, buffer_size, &result](
              auto const first, auto const last)
            {
              auto iter = first;
              for (; iter + buffer_size < last; iter += buffer_size)
              {
                // <Psi_0|Psi_0> = sum_n a_0(n)^* a_0(n) = sum_n |a_0(n)|^2 = 1
                yampi::broadcast(
                  yampi::make_buffer(iter, iter + buffer_size, datatype),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, iter + buffer_size, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient0)
                      { using std::norm; return static_cast<complex_type>(norm(coefficient0)); });
              }

              if (iter < last)
              {
                // <Psi_0|Psi_0> = sum_n a_0(n)^* a_0(n) = sum_n |a_0(n)|^2 = 1
                yampi::broadcast(
                  yampi::make_buffer(iter, last, datatype),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, last, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient0)
                      { using std::norm; return static_cast<complex_type>(norm(coefficient0)); });
              }
            });
        else // if (intercircuit_communicator.rank(environment) == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, &datatype, intercircuit_root, &intercircuit_communicator, &environment,
             buffer_first, buffer_last, buffer_size, &result](
              auto const first, auto const last)
            {
              auto iter = first;
              for (; iter + buffer_size < last; iter += buffer_size)
              {
                // <Psi_k|Psi_0> = sum_n a_k(n)^* a_0(n)
                yampi::broadcast(
                  yampi::make_buffer(buffer_first, buffer_last, datatype),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, iter + buffer_size, buffer_first, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient_k, complex_type const& coefficient0)
                      { using std::conj; return conj(coefficient_k) * coefficient0; });
              }

              if (iter < last)
              {
                // <Psi_k|Psi_0> = sum_n a_k(n)^* a_0(n)
                yampi::broadcast(
                  yampi::make_buffer(buffer_first, buffer_first + (last - iter), datatype),
                  intercircuit_root, intercircuit_communicator, environment);
                result
                  = ::ket::utility::transform_reduce(
                      parallel_policy, iter, last, buffer_first, result,
                      std::plus<complex_type>{},
                      [](complex_type const& coefficient_k, complex_type const& coefficient0)
                      { using std::conj; return conj(coefficient_k) * coefficient0; });
              }
            });

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          circuit_communicator, environment);

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename DerivedDatatype, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, datatype, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          circuit_communicator, environment);

        return result;
      }
    } // namespace inner_product_detail

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    // reduce version
    namespace inner_product_detail
    {
      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          circuit_root, circuit_communicator, environment);

        if (circuit_communicator.rank(environment) != circuit_root)
          return boost::none;

        return result;
      }

      template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename DerivedDatatype, typename BufferAllocator>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, datatype, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          circuit_root, circuit_communicator, environment);

        if (circuit_communicator.rank(environment) != circuit_root)
          return boost::none;

        return result;
      }
    } // namespace inner_product_detail

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename MpiPolicy, typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{::ket::mpi::utility::generate_logger_string(std::string{"Inner product"}), environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    template <typename ParallelPolicy, typename LocalState, typename BufferAllocator, typename DerivedDatatype>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment);
    }

    // <Psi_2| A_{ij} |Psi_1>
    // (1) two states |Psi_1> and |Psi_2> exist in the same MPI group
    namespace inner_product_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1 const& local_state1, LocalState2 const& local_state2,
        yampi::rank const rank_in_unit, Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState1>
      {
        auto const data_block_size
          = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state1, rank_in_unit));
        assert(data_block_size == static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state2, rank_in_unit)));
        auto const num_data_blocks
          = static_cast<StateInteger>(::ket::mpi::utility::policy::num_data_blocks(mpi_policy, rank_in_unit));

        using std::begin;
        using std::end;

        constexpr auto num_operated_qubits = static_cast<BitInteger>(sizeof...(Qubits) + 1u);
# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        auto const num_local_qubits
          = static_cast<BitInteger>(::ket::mpi::utility::policy::num_local_qubits(mpi_policy, data_block_size));
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        std::array<qubit_type, num_operated_qubits + BitInteger{1u}> sorted_qubits_with_sentinel{
          ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...,
          ::ket::make_qubit<StateInteger>(num_local_qubits)};
        std::sort(begin(sorted_qubits_with_sentinel), std::prev(end(sorted_qubits_with_sentinel)));

        std::array<qubit_type, num_operated_qubits> unsorted_qubits{
          ::ket::remove_control(permutated_qubit.qubit()), ::ket::remove_control(permutated_qubits.qubit())...};
# else // KET_USE_BIT_MASKS_EXPLICITLY
        std::array<StateInteger, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks(qubit_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
        std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks(index_masks, permutated_qubit.qubit(), permutated_qubits.qubit()...);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        using complex_type = ::ket::utility::meta::range_value_t<LocalState1>;
        static_assert(
          std::is_same<complex_type, ::ket::utility::meta::range_value_t<LocalState2>>::value,
          "value_type's of LocalState1 and LocalState2 should be the same");
        auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

        auto const local_state_first1 = begin(local_state1);
        auto const local_state_first2 = begin(local_state2);
        for (auto data_block_index = StateInteger{0u}; data_block_index < num_data_blocks; ++data_block_index)
        {
          auto const first1 = local_state_first1 + data_block_index * data_block_size;
          auto const first2 = local_state_first2 + data_block_index * data_block_size;

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::utility::loop_n(
            parallel_policy, data_block_size >> num_operated_qubits,
            [&observable, &partial_sums, first1, first2, unsorted_qubits, sorted_qubits_with_sentinel](
              StateInteger const index_wo_qubits, int const thread_index)
            { partial_sums[thread_index] += observable(first1, first2, index_wo_qubits, unsorted_qubits, sorted_qubits_with_sentinel); });
# else // KET_USE_BIT_MASKS_EXPLICITLY
          ::ket::utility::loop_n(
            parallel_policy, data_block_size >> num_operated_qubits,
            [&observable, &partial_sums, first1, first2, qubit_masks, index_masks](
              StateInteger const index_wo_qubits, int const thread_index)
            { partial_sums[thread_index] += observable(first1, first2, index_wo_qubits, qubit_masks, index_masks); });
# endif // KET_USE_BIT_MASKS_EXPLICITLY
        }

        return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
      }
    } // namespace inner_product_detail

    namespace local
    {
      namespace dispatch
      {
        template <typename LocalState1_, typename LocalState2_>
        struct inner_product
        {
          template <
            typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2,
            typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
          static auto call(
            MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
            LocalState1 const& local_state1, LocalState2 const& local_state2,
            yampi::rank const rank_in_unit, Observable&& observable,
            ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
            ::ket::mpi::permutated<Qubits> const... permutated_qubits)
          -> ::ket::utility::meta::range_value_t<LocalState1>
          {
            assert(::ket::mpi::page::none_on_page(local_state1, permutated_qubit, permutated_qubits...));
            assert(::ket::mpi::page::none_on_page(local_state2, permutated_qubit, permutated_qubits...));
            return ::ket::mpi::inner_product_detail::inner_product(
              mpi_policy, parallel_policy,
              local_state1, local_state2, rank_in_unit,
              std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);
          }
        }; // struct inner_product<LocalState1_, LocalState2_>
      } // namespace dispatch

      template <
        typename MpiPolicy, typename ParallelPolicy, typename LocalState1, typename LocalState2,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1 const& local_state1, LocalState2 const& local_state2,
        yampi::rank const rank_in_unit, Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState1>
      {
        if (::ket::mpi::page::none_on_page(local_state1, permutated_qubit, permutated_qubits...))
        {
          assert(::ket::mpi::page::none_on_page(local_state2, permutated_qubit, permutated_qubits...));

          using inner_product_impl
            = ::ket::mpi::local::dispatch::inner_product<std::remove_cv_t<std::remove_reference_t<LocalState1>>, std::remove_cv_t<std::remove_reference_t<LocalState2>>>;
          inner_product_impl::call(
            mpi_policy, parallel_policy,
            local_state1, local_state2, rank_in_unit,
            std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);
        }

        return ::ket::mpi::inner_product_detail::inner_product(
          mpi_policy, parallel_policy,
          local_state1, local_state2, rank_in_unit,
          std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);
      }
    } // namespace local

    namespace inner_product_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
        typename LocalState2, typename Allocator2, typename BufferAllocator,
        typename Observable, typename... Qubits>
      inline std::enable_if_t<
        ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
        ::ket::utility::meta::range_value_t<LocalState1> >
      inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1& local_state1,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
        LocalState2& local_state2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
        std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        assert(permutation1 == permutation2);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state1, permutation1, buffer, communicator, environment, qubit, qubits...);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state2, permutation2, buffer, communicator, environment, qubit, qubits...);
        assert(permutation1 == permutation2);

        auto result
          = ::ket::mpi::local::inner_product(
              mpi_policy, parallel_policy,
              local_state1, local_state2,
              ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment),
              std::forward<Observable>(observable), permutation1[qubit], permutation1[qubits]...);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          communicator, environment);

        return result;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
        typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
        typename Observable, typename... Qubits>
      inline std::enable_if_t<
        ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
        ::ket::utility::meta::range_value_t<LocalState1> >
      inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1& local_state1,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
        LocalState2& local_state2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
        std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& communicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        assert(permutation1 == permutation2);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state1, permutation1, buffer, datatype, communicator, environment, qubit, qubits...);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state2, permutation2, buffer, datatype, communicator, environment, qubit, qubits...);
        assert(permutation1 == permutation2);

        auto result
          = ::ket::mpi::local::inner_product(
              mpi_policy, parallel_policy,
              local_state1, local_state2,
              ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment),
              std::forward<Observable>(observable), permutation1[qubit], permutation1[qubits]...);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          communicator, environment);

        return result;
      }
    } // namespace inner_product_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState1> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // reduce version
    namespace inner_product_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
        typename LocalState2, typename Allocator2, typename BufferAllocator, typename Observable, typename... Qubits>
      inline std::enable_if_t<
        ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
        boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
      inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1& local_state1,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
        LocalState2& local_state2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
        std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        assert(permutation1 == permutation2);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state1, permutation1, buffer, communicator, environment, qubit, qubits...);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state2, permutation2, buffer, communicator, environment, qubit, qubits...);
        assert(permutation1 == permutation2);

        auto result
          = ::ket::mpi::local::inner_product(
              mpi_policy, parallel_policy,
              local_state1, local_state2,
              ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment),
              std::forward<Observable>(observable), permutation1[qubit], permutation1[qubits]...);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          root, communicator, environment);

        if (communicator.rank(environment) != root)
          return boost::none;

        return result;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
        typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
        typename Observable, typename... Qubits>
      inline std::enable_if_t<
        ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
        boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
      inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState1& local_state1,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
        LocalState2& local_state2,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
        std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      {
        assert(permutation1 == permutation2);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state1, permutation1, buffer, datatype, communicator, environment, qubit, qubits...);
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state2, permutation2, buffer, datatype, communicator, environment, qubit, qubits...);
        assert(permutation1 == permutation2);

        auto result
          = ::ket::mpi::local::inner_product(
              mpi_policy, parallel_policy,
              local_state1, local_state2,
              ::ket::mpi::utility::policy::rank_in_unit(mpi_policy, communicator, environment),
              std::forward<Observable>(observable), permutation1[qubit], permutation1[qubits]...);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          root, communicator, environment);

        if (communicator.rank(environment) != root)
          return boost::none;

        return result;
      }
    } // namespace inner_product_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state1, permutation1, local_state2, permutation2,
        buffer, datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState1>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState1>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState1, typename StateInteger, typename BitInteger, typename Allocator1,
      typename LocalState2, typename Allocator2, typename BufferAllocator, typename DerivedDatatype,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState1> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState1& local_state1,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator1>& permutation1,
      LocalState2& local_state2,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator2>& permutation2,
      std::vector< ::ket::utility::meta::range_value_t<LocalState1>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& communicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state1, permutation1, local_state2, permutation2, buffer, datatype, root, communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // dispatch for <Psi_{remote}| A_{ij} |Psi_{local}> and <Psi_k| A_{ij} |Psi_0>
    namespace dispatch
    {
      template <typename LocalState_>
      struct inner_product_page
      {
        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename LocalState, typename BufferAllocator,
          std::size_t num_operated_qubits,
          typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
        [[noreturn]] static auto call(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          LocalState const& local_state,
          std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
          yampi::rank const rank, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
          std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
          Observable&& observable,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
          ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> ::ket::utility::meta::range_value_t<LocalState>
        { throw 1; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename LocalState, typename BufferAllocator, typename DerivedDatatype,
          std::size_t num_operated_qubits,
          typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
        [[noreturn]] static auto call(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          LocalState const& local_state,
          std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::rank const rank, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
          std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
          Observable&& observable,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
          ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> ::ket::utility::meta::range_value_t<LocalState>
        { throw 1; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename LocalState, typename BufferAllocator,
          std::size_t num_operated_qubits,
          typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
        [[noreturn]] static auto call(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          LocalState const& local_state,
          std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
          yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
          yampi::environment const& environment,
          std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
          Observable&& observable,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
          ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> ::ket::utility::meta::range_value_t<LocalState>
        { throw 1; }

        template <
          typename MpiPolicy, typename ParallelPolicy,
          typename LocalState, typename BufferAllocator, typename DerivedDatatype,
          std::size_t num_operated_qubits,
          typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
        [[noreturn]] static auto call(
          MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
          LocalState const& local_state,
          std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
          yampi::datatype_base<DerivedDatatype> const& datatype,
          yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
          yampi::environment const& environment,
          std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
          Observable&& observable,
          ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
          ::ket::mpi::permutated<Qubits> const... permutated_qubits)
        -> ::ket::utility::meta::range_value_t<LocalState>
        { throw 1; }
      }; // struct inner_product_page<LocalState_>
    } // namespace dispatch

    // <Psi_{remote}| A_{ij} |Psi_{local}>
    // (2) two states |Psi_{local}> and |Psi_{remote}> do not exist in the same MPI group
    // all_reduce version
    namespace inner_product_detail
    {
      template <typename PermutatedQubitIterator, typename StateInteger, typename BitInteger>
      inline auto base_nonbuffer_index(
        PermutatedQubitIterator const permutated_operated_nonbuffer_qubit_first,
        PermutatedQubitIterator const permutated_operated_nonbuffer_qubit_last,
        StateInteger const nonbuffer_index_wo_operated_qubits, BitInteger const num_buffer_qubits)
      -> StateInteger
      {
        using permutated_qubit_type
          = typename std::iterator_traits<PermutatedQubitIterator>::value_type;
        return std::accumulate(
          permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
          nonbuffer_index_wo_operated_qubits,
          [num_buffer_qubits](
            StateInteger const partial_base_nonbuffer_index,
            permutated_qubit_type const permutated_operated_nonbuffer_qubit)
          {
            auto const corrected_permutated_operated_nonbuffer_qubit
              = permutated_operated_nonbuffer_qubit - num_buffer_qubits;
            auto const lower_mask
              = (StateInteger{1u} << corrected_permutated_operated_nonbuffer_qubit)
                  - StateInteger{1u};
            auto const upper_mask = compl lower_mask;
            return ((partial_base_nonbuffer_index bitand upper_mask) << 1)
              bitor (partial_base_nonbuffer_index bitand lower_mask);
          });
      }

      template <
        typename StateInteger, typename PermutatedQubitIterator1,
        typename PermutatedQubitIterator2, typename BitInteger>
      inline auto buffer_aware_index_to_nonbuffer_index(
        StateInteger const buffer_aware_index,
        PermutatedQubitIterator1 const mapped_permutated_buffer_qubit_first,
        PermutatedQubitIterator1 const mapped_permutated_buffer_qubit_last,
        PermutatedQubitIterator2 const permutated_operated_nonbuffer_qubit_first,
        StateInteger const base_nonbuffer_index, BitInteger const num_buffer_qubits)
      -> StateInteger
      {
        using permutated_qubit_type
          = typename std::iterator_traits<PermutatedQubitIterator1>::value_type;
        static_assert(
          std::is_same<permutated_qubit_type, typename std::iterator_traits<PermutatedQubitIterator2>::value_type>::value,
          "The value_type's of PermutatedQubitIteratot1 and PermutatedQubitIterator2 are the same");
        return std::inner_product(
          mapped_permutated_buffer_qubit_first,
          mapped_permutated_buffer_qubit_last,
          permutated_operated_nonbuffer_qubit_first,
          base_nonbuffer_index, std::bit_or<StateInteger>{},
          [buffer_aware_index, num_buffer_qubits](
            permutated_qubit_type const mapped_permutated_buffer_qubit,
            permutated_qubit_type const permutated_operated_nonbuffer_qubit)
          {
            return
              ((buffer_aware_index bitand (StateInteger{1u} << mapped_permutated_buffer_qubit))
                 >> mapped_permutated_buffer_qubit)
                << (permutated_operated_nonbuffer_qubit - num_buffer_qubits);
          });
      }

      template <typename StateInteger, typename PermutatedQubitIterator, typename BitInteger>
      inline auto buffer_index(
        PermutatedQubitIterator const mapped_permutated_buffer_qubit_first,
        PermutatedQubitIterator const mapped_permutated_buffer_qubit_last,
        StateInteger const buffer_aware_index, StateInteger const mapped_buffer_qubits_bits,
        BitInteger const num_buffer_qubits)
      -> StateInteger
      {
        auto result = buffer_aware_index;

        for (auto mapped_permutated_buffer_qubit_iter = mapped_permutated_buffer_qubit_first;
             mapped_permutated_buffer_qubit_iter != mapped_permutated_buffer_qubit_last;
             ++mapped_permutated_buffer_qubit_iter)
        {
          auto const iter_index
            = mapped_permutated_buffer_qubit_iter - mapped_permutated_buffer_qubit_first;
          result
            = (result bitand (compl (StateInteger{1u} << *mapped_permutated_buffer_qubit_iter)))
                bitor (((mapped_buffer_qubits_bits bitand (StateInteger{1u} << iter_index)) >> iter_index) << *mapped_permutated_buffer_qubit_iter);
        }

        return result;
      }

      template <typename PermutatedQubitIterator, typename StateInteger, typename BitInteger>
      inline auto generate_mapped_permutated_buffer_qubits(
        PermutatedQubitIterator const permutated_operated_buffer_qubit_first,
        PermutatedQubitIterator const permutated_operated_buffer_qubit_last,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const least_permutated_nonbuffer_qubit,
        BitInteger const num_operated_nonbuffer_qubits)
      -> std::vector< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > >
      {
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
        auto result = std::vector<permutated_qubit_type>(num_operated_nonbuffer_qubits);

        auto possible_mapped_permutated_buffer_qubit = least_permutated_nonbuffer_qubit;
        auto permutated_operated_buffer_qubit_iter = permutated_operated_buffer_qubit_last;

        using std::rbegin;
        using std::rend;
        auto const rlast = rend(result);
        for (auto riter = rbegin(result); riter != rlast; ++riter)
        {
          --possible_mapped_permutated_buffer_qubit;
          if (possible_mapped_permutated_buffer_qubit < *permutated_operated_buffer_qubit_first)
          {
            *riter = possible_mapped_permutated_buffer_qubit;
            continue;
          }

          if (permutated_operated_buffer_qubit_iter != permutated_operated_buffer_qubit_first)
            --permutated_operated_buffer_qubit_iter;

          while (possible_mapped_permutated_buffer_qubit == *permutated_operated_buffer_qubit_iter)
          {
            --possible_mapped_permutated_buffer_qubit;

            if (permutated_operated_buffer_qubit_iter == permutated_operated_buffer_qubit_first)
              break;

            --permutated_operated_buffer_qubit_iter;
          }

          *riter = possible_mapped_permutated_buffer_qubit;
        }

        return result;
      }

      struct buffer_aware_sentinel { };

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      class buffer_aware_iterator
      {
        using permutated_qubit_type = typename std::iterator_traits<PermutatedQubitIterator1>::value_type;
        static_assert(
          std::is_same<permutated_qubit_type, typename std::iterator_traits<PermutatedQubitIterator2>::value_type>::value,
          "The value_type's of PermutatedQubitIterator1 and PermutatedQubitIterator2 are the same");
        using state_integer_type = ::ket::meta::state_integer_t<permutated_qubit_type>;
        using bit_integer_type = ::ket::meta::bit_integer_t<permutated_qubit_type>;

       public:
        using value_type = typename std::iterator_traits<StateIterator>::value_type;
        using difference_type = typename std::iterator_traits<StateIterator>::difference_type;
        using pointer = typename std::iterator_traits<StateIterator>::pointer;
        using reference = typename std::iterator_traits<StateIterator>::reference;
        using iterator_category = typename std::iterator_traits<StateIterator>::iterator_category;

       private:
        StateIterator state_first_;
        PermutatedQubitIterator1 mapped_permutated_buffer_qubit_first_;
        PermutatedQubitIterator1 mapped_permutated_buffer_qubit_last_;
        PermutatedQubitIterator2 permutated_operated_nonbuffer_qubit_first_;
        state_integer_type base_nonbuffer_index_;
        state_integer_type mapped_buffer_qubit_bits_;
        bit_integer_type num_buffer_qubits_;

        difference_type index_;

       public:
        template <typename StateInteger, typename BitInteger>
        buffer_aware_iterator(
          StateIterator const state_first,
          PermutatedQubitIterator1 const mapped_permutated_buffer_qubit_first,
          PermutatedQubitIterator1 const mapped_permutated_buffer_qubit_last,
          PermutatedQubitIterator2 const permutated_operated_nonbuffer_qubit_first,
          PermutatedQubitIterator2 const permutated_operated_nonbuffer_qubit_last,
          StateInteger const nonbuffer_index_wo_operated_qubits,
          StateInteger const mapped_buffer_qubit_bits,
          BitInteger const num_buffer_qubits,
          difference_type const index = difference_type{0}) noexcept
          : state_first_{state_first},
            mapped_permutated_buffer_qubit_first_{mapped_permutated_buffer_qubit_first},
            mapped_permutated_buffer_qubit_last_{mapped_permutated_buffer_qubit_last},
            permutated_operated_nonbuffer_qubit_first_{permutated_operated_nonbuffer_qubit_first},
            base_nonbuffer_index_{
              ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                nonbuffer_index_wo_operated_qubits, num_buffer_qubits)},
            mapped_buffer_qubit_bits_{mapped_buffer_qubit_bits},
            num_buffer_qubits_{num_buffer_qubits},
            index_{index}
        {
          static_assert(
            std::is_same<StateInteger, state_integer_type>::value,
            "StateInteger should be the same as state_integer_type of value_type of PermutatedQubitIterator1");
          static_assert(
            std::is_same<BitInteger, bit_integer_type>::value,
            "BitInteger should be the same as bit_integer_type of value_type of PermutatedQubitIterator1");
          assert(
            mapped_permutated_buffer_qubit_last - mapped_permutated_buffer_qubit_first
            == permutated_operated_nonbuffer_qubit_last - permutated_operated_nonbuffer_qubit_first);
        }

        auto operator==(buffer_aware_iterator const& other) const noexcept -> bool
        {
          assert(state_first_ == other.state_first_);
          assert(mapped_permutated_buffer_qubit_first_ == other.mapped_permutated_buffer_qubit_firststate_first_);
          assert(mapped_permutated_buffer_qubit_last_ == other.mapped_permutated_buffer_qubit_firststate_last_);
          assert(permutated_operated_nonbuffer_qubit_first_ == other.permutated_operated_nonbuffer_qubit_first_);
          assert(base_nonbuffer_index_ == other.base_nonbuffer_index_);
          assert(mapped_buffer_qubit_bits_ == other.mapped_buffer_qubit_bits_);
          assert(num_buffer_qubits_ == other.num_buffer_qubits_);
          return index_ == other.index_;
        }

        auto operator<(buffer_aware_iterator const& other) const noexcept -> bool
        {
          assert(state_first_ == other.state_first_);
          assert(mapped_permutated_buffer_qubit_first_ == other.mapped_permutated_buffer_qubit_firststate_first_);
          assert(mapped_permutated_buffer_qubit_last_ == other.mapped_permutated_buffer_qubit_firststate_last_);
          assert(permutated_operated_nonbuffer_qubit_first_ == other.permutated_operated_nonbuffer_qubit_first_);
          assert(base_nonbuffer_index_ == other.base_nonbuffer_index_);
          assert(mapped_buffer_qubit_bits_ == other.mapped_buffer_qubit_bits_);
          assert(num_buffer_qubits_ == other.num_buffer_qubits_);
          return index_ < other.index_;
        }

        auto operator==(::ket::mpi::inner_product_detail::buffer_aware_sentinel const& other) const noexcept -> bool
        { return index_ == ::ket::utility::integer_exp2<state_integer_type>(num_buffer_qubits_); }

        auto operator<(::ket::mpi::inner_product_detail::buffer_aware_sentinel const& other) const noexcept -> bool
        { return index_ < ::ket::utility::integer_exp2<state_integer_type>(num_buffer_qubits_); }

        auto operator*() const noexcept -> reference
        {
          auto const nonbuffer_index
            = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                static_cast<state_integer_type>(index_),
                mapped_permutated_buffer_qubit_first_, mapped_permutated_buffer_qubit_last_,
                permutated_operated_nonbuffer_qubit_first_,
                base_nonbuffer_index_, num_buffer_qubits_);

          auto const buffer_index
            = ::ket::mpi::inner_product_detail::buffer_index(
                mapped_permutated_buffer_qubit_first_, mapped_permutated_buffer_qubit_last_,
                static_cast<state_integer_type>(index_), mapped_buffer_qubit_bits_,
                num_buffer_qubits_);

          return *(state_first_ + ((nonbuffer_index << num_buffer_qubits_) + buffer_index));
        }

        auto operator[](difference_type const n) const -> reference
        {
          auto const index = static_cast<state_integer_type>(index_ + n);

          auto const nonbuffer_index
            = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                index, mapped_permutated_buffer_qubit_first_, mapped_permutated_buffer_qubit_last_,
                permutated_operated_nonbuffer_qubit_first_,
                base_nonbuffer_index_, num_buffer_qubits_);

          auto const buffer_index
            = ::ket::mpi::inner_product_detail::buffer_index(
                mapped_permutated_buffer_qubit_first_, mapped_permutated_buffer_qubit_last_,
                index, mapped_buffer_qubit_bits_, num_buffer_qubits_);

          return *(state_first_ + ((nonbuffer_index << num_buffer_qubits_) + buffer_index));
        }

        auto operator++() noexcept -> buffer_aware_iterator& { ++index_; return *this; }
        auto operator++(int) noexcept -> buffer_aware_iterator { auto result = *this; ++*this; return result; }
        auto operator--() noexcept -> buffer_aware_iterator& { --index_; return *this; }
        auto operator--(int) noexcept -> buffer_aware_iterator { auto result = *this; --*this; return result; }
        auto operator+=(difference_type const n) noexcept -> buffer_aware_iterator& { index_ += n; return *this; }
        auto operator-=(difference_type const n) noexcept -> buffer_aware_iterator& { index_ -= n; return *this; }
        auto operator-(buffer_aware_iterator const& other) const noexcept -> difference_type { return index_ - other.index_; }

        auto swap(buffer_aware_iterator& other) noexcept -> void
        {
          using std::swap;
          swap(state_first_, other.state_first_);
          swap(mapped_permutated_buffer_qubit_first_, other.mapped_permutated_buffer_qubit_first_);
          swap(mapped_permutated_buffer_qubit_last_, other.mapped_permutated_buffer_qubit_last_);
          swap(permutated_operated_nonbuffer_qubit_first_, other.permutated_operated_nonbuffer_qubit_first_);
          swap(base_nonbuffer_index_, other.base_nonbuffer_index_);
          swap(mapped_buffer_qubit_bits_, other.mapped_buffer_qubit_bits_);
          swap(num_buffer_qubits_, other.num_buffer_qubits_);
          swap(index_, other.index_);
        }
      }; // class buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator!=(
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return not (lhs == rhs); }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator>(
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return rhs < lhs; }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator<=(
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return not (lhs > rhs); }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator>=(
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& lhs,
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> const& rhs)
      -> bool
      { return not (lhs < rhs); }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator+(
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> iter,
        typename ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>::difference_type const n)
      -> ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>
      { return iter += n; }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator+(
        typename ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>::difference_type const n,
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> iter)
      -> ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>
      { return iter += n; }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto operator-(
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2> iter,
        typename ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>::difference_type const n)
      -> ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>
      { return iter -= n; }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto swap(
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>& lhs,
        ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>& rhs) noexcept
      -> void
      { lhs.swap(rhs); }

      template <typename StateIterator, typename PermutatedQubitIterator1, typename PermutatedQubitIterator2>
      inline auto make_buffer_aware_iterator(
        StateIterator const state_first,
        PermutatedQubitIterator1 const mapped_permutated_buffer_qubit_first,
        PermutatedQubitIterator1 const mapped_permutated_buffer_qubit_last,
        PermutatedQubitIterator2 const permutated_operated_nonbuffer_qubit_first,
        PermutatedQubitIterator2 const permutated_operated_nonbuffer_qubit_last,
        ::ket::meta::state_integer_t<typename std::iterator_traits<PermutatedQubitIterator1>::value_type> const nonbuffer_index_wo_operated_qubits,
        ::ket::meta::state_integer_t<typename std::iterator_traits<PermutatedQubitIterator1>::value_type> const mapped_buffer_qubit_bits,
        ::ket::meta::bit_integer_t<typename std::iterator_traits<PermutatedQubitIterator1>::value_type> const num_buffer_qubits,
        typename std::iterator_traits<PermutatedQubitIterator1>::difference_type const index
          = typename std::iterator_traits<PermutatedQubitIterator1>::difference_type{0u}) noexcept
      -> ::ket::mpi::inner_product_detail::buffer_aware_iterator<StateIterator, PermutatedQubitIterator1, PermutatedQubitIterator2>
      {
        return {state_first,
          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
          permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
          nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits, index};
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator, std::size_t num_operated_qubits,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl_p0(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        static_assert(num_operated_qubits == sizeof...(Qubits) + 1u, "The number of permutated_qubit's is the same as num_operated_qubits");

        auto const rank = intracommunicator.rank(environment);
        assert(rank == intercommunicator.rank(environment));
# ifndef NDEBUG
        auto const num_processes = intracommunicator.size(environment);
# endif // NDEBUG
        assert(num_processes == intercommunicator.size(environment) and num_processes == intercommunicator.remote_size(environment));

        // nonbuffer buffer
        //   xxxxx|xxxxxxxxxx
        //    ^ ^  ^ ^   ^    <- operated qubits
        //   n n n            <- nonbuffer_index_wo_operated_qubits
        //          * *       <- mapped_buffer_qubit_bits
        //    @ @             <- operated_nonbuffer_qubit_bits
        //         u u dddddd <- "buffer_index_wo_mapped_qubits"
        //         u u        <- upper_buffer_index_wo_mapped_qubits
        //             dddddd <- "lower_buffer_index_wo_mapped_qubits"
        //    @ @  u*u*dddddd <- "buffer_aware_index" (** => @@)

        // buffer_first, buffer_last, buffer_size, num_buffer_qubits
        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const num_buffer_qubits
          = ::ket::utility::integer_log2<BitInteger>(
              static_cast<StateInteger>(::ket::mpi::utility::buffer_end(local_state, buffer) - buffer_first));
        auto const buffer_size = ::ket::utility::integer_exp2<StateInteger>(num_buffer_qubits);
        // nonbuffer_size
        auto const nonbuffer_size
          = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment))
            / buffer_size;

        // least_permuated_nonbuffer_qubit
        auto const least_permutated_nonbuffer_qubit
          = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(num_buffer_qubits));

        // permutated_operated_buffer_qubits, permutated_operated_nonbuffer_qubits
        using std::begin;
        using std::end;
        auto const permutated_operated_buffer_qubit_first = begin(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_last = end(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_first
          = std::lower_bound(
              begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array),
              least_permutated_nonbuffer_qubit);
        auto const permutated_operated_buffer_qubit_last = permutated_operated_nonbuffer_qubit_first;
        // num_operated_buffer_qubits, num_operated_nonbuffer_qubits
# ifndef NDEBUG
        auto const num_operated_buffer_qubits
          = static_cast<BitInteger>(permutated_operated_buffer_qubit_last - permutated_operated_buffer_qubit_first);
# endif // NDEBUG
        auto const num_operated_nonbuffer_qubits
          = static_cast<BitInteger>(permutated_operated_nonbuffer_qubit_last - permutated_operated_nonbuffer_qubit_first);
        assert(num_operated_buffer_qubits + num_operated_nonbuffer_qubits == num_operated_qubits);
        // num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values
        auto const num_nonbuffer_indices_wo_operated_qubits = nonbuffer_size >> num_operated_nonbuffer_qubits;
        auto const num_operated_nonbuffer_qubit_values = ::ket::utility::integer_exp2<StateInteger>(num_operated_nonbuffer_qubits);

        // mapped_permutated_buffer_qubits
        auto const mapped_permutated_buffer_qubits
          = ::ket::mpi::inner_product_detail::generate_mapped_permutated_buffer_qubits(
              permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
              least_permutated_nonbuffer_qubit, num_operated_nonbuffer_qubits);
        auto const mapped_permutated_buffer_qubit_first = begin(mapped_permutated_buffer_qubits);
        auto const mapped_permutated_buffer_qubit_last = end(mapped_permutated_buffer_qubits);
        // num_lower_buffer_indices
        auto const num_lower_buffer_indices = mapped_permutated_buffer_qubits.empty() ? buffer_size : StateInteger{1u} << *mapped_permutated_buffer_qubit_first;

        // main loop
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        std::array<qubit_type, num_operated_qubits> modified_unsorted_qubits{
          ::ket::remove_control(permutated_qubit.qubit()),
          ::ket::remove_control(permutated_qubits.qubit())...};
        auto mapped_permutated_buffer_qubit_iter = mapped_permutated_buffer_qubit_first;
        for (auto permutated_operated_nonbuffer_qubit_iter = permutated_operated_nonbuffer_qubit_first;
             permutated_operated_nonbuffer_qubit_iter != permutated_operated_nonbuffer_qubit_last;
             ++permutated_operated_nonbuffer_qubit_iter, ++mapped_permutated_buffer_qubit_iter)
        {
          auto const found
            = std::find(
                begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
                permutated_operated_nonbuffer_qubit_iter->qubit());
          if (found != end(modified_unsorted_qubits))
            *found = mapped_permutated_buffer_qubit_iter->qubit();
        }

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        std::array<qubit_type, num_operated_qubits + 1u> modified_sorted_qubits_with_sentinel{ };
        std::copy(
          begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
          begin(modified_sorted_qubits_with_sentinel));
        modified_sorted_qubits_with_sentinel.back()
          = ::ket::make_qubit<StateInteger>(num_buffer_qubits);
        std::sort(
          begin(modified_sorted_qubits_with_sentinel),
          std::prev(end(modified_sorted_qubits_with_sentinel)));
# else // KET_USE_BIT_MASKS_EXPLICITLY
        std::array<StateInteger, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks_from_tuple(modified_unsorted_qubits, qubit_masks);
        std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks_from_tuple(modified_unsorted_qubits, index_masks);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, intracommunicator, environment,
          [parallel_policy, &intercommunicator, &environment, &observable,
           rank, buffer_first, num_buffer_qubits, buffer_size,
           permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
           permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
           num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
           mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
           num_lower_buffer_indices,
           &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums](
            auto const first, auto const last)
          {
            // nnn
            for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                 nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                 ++nonbuffer_index_wo_operated_qubits)
            {
              // n0n0n
              auto const base_nonbuffer_index
                = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                    permutated_operated_nonbuffer_qubit_first,
                    permutated_operated_nonbuffer_qubit_last,
                    nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

              // **
              for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                   mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                   ++mapped_buffer_qubit_bits)
              {
                auto buffer_iter = buffer_first;

                for (auto buffer_aware_first_index = StateInteger{0u};
                     buffer_aware_first_index < buffer_size;
                     buffer_aware_first_index += num_lower_buffer_indices,
                     buffer_iter += num_lower_buffer_indices)
                {
                  auto const nonbuffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                        buffer_aware_first_index,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first,
                        base_nonbuffer_index, num_buffer_qubits);

                  auto const buffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_index(
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        buffer_aware_first_index, mapped_buffer_qubit_bits,
                        num_buffer_qubits);

                  auto const chunk_first
                    = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                  auto const tag = yampi::tag{rank.mpi_rank()};
                  yampi::send_receive(
                    yampi::ignore_status,
                    yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices), rank, tag,
                    yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices), rank, tag,
                    intercommunicator, environment);
                }

                auto const buffer_aware_first
                  = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                      first,
                      mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                      permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                ::ket::utility::loop_n(
                  parallel_policy, buffer_size >> num_operated_qubits,
                  [&observable, buffer_first, &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums, buffer_aware_first](
                    StateInteger const index_wo_qubits, int const thread_index)
                  {
                    partial_sums[thread_index]
                      += observable(
                           buffer_aware_first, buffer_first, index_wo_qubits,
                           modified_unsorted_qubits, modified_sorted_qubits_with_sentinel);
                  });
              }
            }
          });
# else // KET_USE_BIT_MASKS_EXPLICITLY
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, intracommunicator, environment,
          [parallel_policy, &intercommunicator, &environment, &observable,
           rank, buffer_first, num_buffer_qubits, buffer_size,
           permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
           permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
           num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
           mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
           num_lower_buffer_indices,
           &qubit_masks, &index_masks, &partial_sums](
            auto const first, auto const last)
          {
            // nnn
            for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                 nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                 ++nonbuffer_index_wo_operated_qubits)
            {
              // n0n0n
              auto const base_nonbuffer_index
                = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                    permutated_operated_nonbuffer_qubit_first,
                    permutated_operated_nonbuffer_qubit_last,
                    nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

              // **
              for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                   mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                   ++mapped_buffer_qubit_bits)
              {
                auto buffer_iter = buffer_first;

                for (auto buffer_aware_first_index = StateInteger{0u};
                     buffer_aware_first_index < buffer_size;
                     buffer_aware_first_index += num_lower_buffer_indices,
                     buffer_iter += num_lower_buffer_indices)
                {
                  auto const nonbuffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                        buffer_aware_first_index,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first,
                        base_nonbuffer_index, num_buffer_qubits);

                  auto const buffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_index(
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        buffer_aware_first_index, mapped_buffer_qubit_bits,
                        num_buffer_qubits);

                  auto const chunk_first
                    = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                  auto const tag = yampi::tag{rank.mpi_rank()};
                  yampi::send_receive(
                    yampi::ignore_status,
                    yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices), rank, tag,
                    yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices), rank, tag,
                    intercommunicator, environment);
                }

                auto const buffer_aware_first
                  = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                      first,
                      mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                      permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                ::ket::utility::loop_n(
                  parallel_policy, buffer_size >> num_operated_qubits,
                  [&observable, buffer_first, &qubit_masks, &index_masks, &partial_sums, buffer_aware_first](
                    StateInteger const index_wo_qubits, int const thread_index)
                  {
                    partial_sums[thread_index]
                      += observable(
                           buffer_aware_first, buffer_first, index_wo_qubits,
                           qubit_masks, index_masks);
                  });
              }
            }
          });
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator, typename DerivedDatatype, std::size_t num_operated_qubits,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl_p0(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        static_assert(num_operated_qubits == sizeof...(Qubits) + 1u, "The number of permutated_qubit's is the same as num_operated_qubits");

        auto const rank = intracommunicator.rank(environment);
        assert(rank == intercommunicator.rank(environment));
# ifndef NDEBUG
        auto const num_processes = intracommunicator.size(environment);
# endif // NDEBUG
        assert(num_processes == intercommunicator.size(environment) and num_processes == intercommunicator.remote_size(environment));

        // nonbuffer buffer
        //   xxxxx|xxxxxxxxxx
        //    ^ ^  ^ ^   ^    <- operated qubits
        //   n n n            <- nonbuffer_index_wo_operated_qubits
        //          * *       <- mapped_buffer_qubit_bits
        //    @ @             <- operated_nonbuffer_qubit_bits
        //         u u dddddd <- "buffer_index_wo_mapped_qubits"
        //         u u        <- upper_buffer_index_wo_mapped_qubits
        //             dddddd <- "lower_buffer_index_wo_mapped_qubits"
        //    @ @  u*u*dddddd <- "buffer_aware_index" (** => @@)

        // buffer_first, buffer_last, buffer_size, num_buffer_qubits
        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const num_buffer_qubits
          = ::ket::utility::integer_log2<BitInteger>(
              static_cast<StateInteger>(::ket::mpi::utility::buffer_end(local_state, buffer) - buffer_first));
        auto const buffer_size = ::ket::utility::integer_exp2<StateInteger>(num_buffer_qubits);
        // nonbuffer_size
        auto const nonbuffer_size
          = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment))
            / buffer_size;

        // least_permuated_nonbuffer_qubit
        auto const least_permutated_nonbuffer_qubit
          = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(num_buffer_qubits));

        // permutated_operated_buffer_qubits, permutated_operated_nonbuffer_qubits
        using std::begin;
        using std::end;
        auto const permutated_operated_buffer_qubit_first = begin(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_last = end(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_first
          = std::lower_bound(
              begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array),
              least_permutated_nonbuffer_qubit);
        auto const permutated_operated_buffer_qubit_last = permutated_operated_nonbuffer_qubit_first;
        // num_operated_buffer_qubits, num_operated_nonbuffer_qubits
# ifndef NDEBUG
        auto const num_operated_buffer_qubits
          = static_cast<BitInteger>(permutated_operated_buffer_qubit_last - permutated_operated_buffer_qubit_first);
# endif // NDEBUG
        auto const num_operated_nonbuffer_qubits
          = static_cast<BitInteger>(permutated_operated_nonbuffer_qubit_last - permutated_operated_nonbuffer_qubit_first);
        assert(num_operated_buffer_qubits + num_operated_nonbuffer_qubits == num_operated_qubits);
        // num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values
        auto const num_nonbuffer_indices_wo_operated_qubits = nonbuffer_size >> num_operated_nonbuffer_qubits;
        auto const num_operated_nonbuffer_qubit_values = ::ket::utility::integer_exp2<StateInteger>(num_operated_nonbuffer_qubits);

        // mapped_permutated_buffer_qubits
        auto const mapped_permutated_buffer_qubits
          = ::ket::mpi::inner_product_detail::generate_mapped_permutated_buffer_qubits(
              permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
              least_permutated_nonbuffer_qubit, num_operated_nonbuffer_qubits);
        auto const mapped_permutated_buffer_qubit_first = begin(mapped_permutated_buffer_qubits);
        auto const mapped_permutated_buffer_qubit_last = end(mapped_permutated_buffer_qubits);
        // num_lower_buffer_indices
        auto const num_lower_buffer_indices = mapped_permutated_buffer_qubits.empty() ? buffer_size : StateInteger{1u} << *mapped_permutated_buffer_qubit_first;

        // main loop
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        std::array<qubit_type, num_operated_qubits> modified_unsorted_qubits{
          ::ket::remove_control(permutated_qubit.qubit()),
          ::ket::remove_control(permutated_qubits.qubit())...};
        auto mapped_permutated_buffer_qubit_iter = mapped_permutated_buffer_qubit_first;
        for (auto permutated_operated_nonbuffer_qubit_iter = permutated_operated_nonbuffer_qubit_first;
             permutated_operated_nonbuffer_qubit_iter != permutated_operated_nonbuffer_qubit_last;
             ++permutated_operated_nonbuffer_qubit_iter, ++mapped_permutated_buffer_qubit_iter)
        {
          auto const found
            = std::find(
                begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
                permutated_operated_nonbuffer_qubit_iter->qubit());
          if (found != end(modified_unsorted_qubits))
            *found = mapped_permutated_buffer_qubit_iter->qubit();
        }

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        std::array<qubit_type, num_operated_qubits + 1u> modified_sorted_qubits_with_sentinel{ };
        std::copy(
          begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
          begin(modified_sorted_qubits_with_sentinel));
        modified_sorted_qubits_with_sentinel.back()
          = ::ket::make_qubit<StateInteger>(num_buffer_qubits);
        std::sort(
          begin(modified_sorted_qubits_with_sentinel),
          std::prev(end(modified_sorted_qubits_with_sentinel)));
# else // KET_USE_BIT_MASKS_EXPLICITLY
        std::array<StateInteger, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks_from_tuple(modified_unsorted_qubits, qubit_masks);
        std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks_from_tuple(modified_unsorted_qubits, index_masks);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, intracommunicator, environment,
          [parallel_policy, &datatype, &intercommunicator, &environment, &observable,
           rank, buffer_first, num_buffer_qubits, buffer_size,
           permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
           permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
           num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
           mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
           num_lower_buffer_indices,
           &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums](
            auto const first, auto const last)
          {
            // nnn
            for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                 nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                 ++nonbuffer_index_wo_operated_qubits)
            {
              // n0n0n
              auto const base_nonbuffer_index
                = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                    permutated_operated_nonbuffer_qubit_first,
                    permutated_operated_nonbuffer_qubit_last,
                    nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

              // **
              for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                   mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                   ++mapped_buffer_qubit_bits)
              {
                auto buffer_iter = buffer_first;

                for (auto buffer_aware_first_index = StateInteger{0u};
                     buffer_aware_first_index < buffer_size;
                     buffer_aware_first_index += num_lower_buffer_indices,
                     buffer_iter += num_lower_buffer_indices)
                {
                  auto const nonbuffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                        buffer_aware_first_index,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first,
                        base_nonbuffer_index, num_buffer_qubits);

                  auto const buffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_index(
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        buffer_aware_first_index, mapped_buffer_qubit_bits,
                        num_buffer_qubits);

                  auto const chunk_first
                    = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                  auto const tag = yampi::tag{rank.mpi_rank()};
                  yampi::send_receive(
                    yampi::ignore_status,
                    yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices, datatype), rank, tag,
                    yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices, datatype), rank, tag,
                    intercommunicator, environment);
                }

                auto const buffer_aware_first
                  = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                      first,
                      mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                      permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                ::ket::utility::loop_n(
                  parallel_policy, buffer_size >> num_operated_qubits,
                  [&observable, buffer_first, &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums, buffer_aware_first](
                    StateInteger const index_wo_qubits, int const thread_index)
                  {
                    partial_sums[thread_index]
                      += observable(
                           buffer_aware_first, buffer_first, index_wo_qubits,
                           modified_unsorted_qubits, modified_sorted_qubits_with_sentinel);
                  });
              }
            }
          });
# else // KET_USE_BIT_MASKS_EXPLICITLY
        ::ket::mpi::utility::for_each_local_range(
          mpi_policy, local_state, intracommunicator, environment,
          [parallel_policy, &datatype, &intercommunicator, &environment, &observable,
           rank, buffer_first, num_buffer_qubits, buffer_size,
           permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
           permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
           num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
           mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
           num_lower_buffer_indices,
           &qubit_masks, &index_masks, &partial_sums](
            auto const first, auto const last)
          {
            // nnn
            for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                 nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                 ++nonbuffer_index_wo_operated_qubits)
            {
              // n0n0n
              auto const base_nonbuffer_index
                = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                    permutated_operated_nonbuffer_qubit_first,
                    permutated_operated_nonbuffer_qubit_last,
                    nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

              // **
              for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                   mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                   ++mapped_buffer_qubit_bits)
              {
                auto buffer_iter = buffer_first;

                for (auto buffer_aware_first_index = StateInteger{0u};
                     buffer_aware_first_index < buffer_size;
                     buffer_aware_first_index += num_lower_buffer_indices,
                     buffer_iter += num_lower_buffer_indices)
                {
                  auto const nonbuffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                        buffer_aware_first_index,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first,
                        base_nonbuffer_index, num_buffer_qubits);

                  auto const buffer_first_index
                    = ::ket::mpi::inner_product_detail::buffer_index(
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        buffer_aware_first_index, mapped_buffer_qubit_bits,
                        num_buffer_qubits);

                  auto const chunk_first
                    = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                  auto const tag = yampi::tag{rank.mpi_rank()};
                  yampi::send_receive(
                    yampi::ignore_status,
                    yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices, datatype), rank, tag,
                    yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices, datatype), rank, tag,
                    intercommunicator, environment);
                }

                auto const buffer_aware_first
                  = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                      first,
                      mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                      permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                ::ket::utility::loop_n(
                  parallel_policy, buffer_size >> num_operated_qubits,
                  [&observable, buffer_first, &qubit_masks, &index_masks, &partial_sums, buffer_aware_first](
                    StateInteger const index_wo_qubits, int const thread_index)
                  {
                    partial_sums[thread_index]
                      += observable(
                           buffer_aware_first, buffer_first, index_wo_qubits,
                           qubit_masks, index_masks);
                  });
              }
            }
          });
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          ::ket::utility::all_in_state_vector(
            static_cast<BitInteger>(
              ::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, intracommunicator, environment)),
            permutated_qubit.qubit(), permutated_qubits.qubit()...));

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
        constexpr auto num_operated_qubits = BitInteger{sizeof...(Qubits) + 1u};
        std::array<permutated_qubit_type, num_operated_qubits> sorted_permutated_operated_qubits_array{permutated_qubit, permutated_qubits...};
        using std::begin;
        using std::end;
        std::sort(begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array));

        if (::ket::mpi::page::is_on_page(sorted_permutated_operated_qubits_array.back(), local_state))
          return ::ket::mpi::dispatch::inner_product_page<std::remove_const_t<std::remove_reference_t<LocalState>>>::call(
            mpi_policy, parallel_policy,
            local_state, buffer,
            intracommunicator.rank(environment), intercommunicator, environment,
            sorted_permutated_operated_qubits_array,
            std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);

        return ::ket::mpi::inner_product_detail::inner_product_impl_p0(
          mpi_policy, parallel_policy,
          local_state, buffer, intracommunicator, intercommunicator, environment,
          sorted_permutated_operated_qubits_array,
          std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator, typename DerivedDatatype,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          ::ket::utility::all_in_state_vector(
            static_cast<BitInteger>(
              ::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, intracommunicator, environment)),
            permutated_qubit.qubit(), permutated_qubits.qubit()...));

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
        constexpr auto num_operated_qubits = BitInteger{sizeof...(Qubits) + 1u};
        std::array<permutated_qubit_type, num_operated_qubits> sorted_permutated_operated_qubits_array{permutated_qubit, permutated_qubits...};
        std::sort(begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array));

        if (::ket::mpi::page::is_on_page(sorted_permutated_operated_qubits_array.back(), local_state))
          return ::ket::mpi::dispatch::inner_product_page<std::remove_const_t<std::remove_reference_t<LocalState>>>::call(
            mpi_policy, parallel_policy,
            local_state, buffer, datatype,
            intracommunicator.rank(environment), intercommunicator, environment,
            sorted_permutated_operated_qubits_array,
            std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);

        return ::ket::mpi::inner_product_detail::inner_product_impl_p0(
          mpi_policy, parallel_policy,
          local_state, buffer, datatype, intracommunicator, intercommunicator, environment,
          sorted_permutated_operated_qubits_array,
          std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, intracommunicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, intracommunicator, intercommunicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          intracommunicator, environment);

        return result;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype, typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, intracommunicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, datatype, intracommunicator, intercommunicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          intracommunicator, environment);

        return result;
      }
    } // namespace inner_product_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // reduce version
    namespace inner_product_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(intracommunicator.size(environment) == intercommunicator.size(environment));

        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, intracommunicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy, local_state, buffer, intercommunicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          root, intracommunicator, environment);

        if (intracommunicator.rank(environment) != root)
          return boost::none;

        return result;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype, typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
        Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(intracommunicator.size(environment) == intercommunicator.size(environment));

        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, intracommunicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, intracommunicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy, local_state, buffer, datatype, intercommunicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          root, intracommunicator, environment);

        if (intracommunicator.rank(environment) != root)
          return boost::none;

        return result;
      }
    } // namespace inner_product_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const root, yampi::communicator const& intracommunicator, yampi::intercommunicator const& intercommunicator, yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, root, intracommunicator, intercommunicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // <Psi_k| A_{ij} |Psi_0> (k = 0, ..., N_{states}: value of intercircuit_communicator.rank(environment))
    // (3) states |Psi_k> do not exist in the same MPI group
    // all_reduce version
    namespace inner_product_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator, std::size_t num_operated_qubits,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl_p0(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        static_assert(num_operated_qubits == sizeof...(Qubits) + 1u, "The number of permutated_qubit's is the same as num_operated_qubits");

        // nonbuffer buffer
        //   xxxxx|xxxxxxxxxx
        //    ^ ^  ^ ^   ^    <- operated qubits
        //   n n n            <- nonbuffer_index_wo_operated_qubits
        //          * *       <- mapped_buffer_qubit_bits
        //    @ @             <- operated_nonbuffer_qubit_bits
        //         u u dddddd <- "buffer_index_wo_mapped_qubits"
        //         u u        <- upper_buffer_index_wo_mapped_qubits
        //             dddddd <- "lower_buffer_index_wo_mapped_qubits"
        //    @ @  u*u*dddddd <- "buffer_aware_index" (** => @@)

        // buffer_first, buffer_last, buffer_size, num_buffer_qubits
        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const num_buffer_qubits
          = ::ket::utility::integer_log2<BitInteger>(
              static_cast<StateInteger>(::ket::mpi::utility::buffer_end(local_state, buffer) - buffer_first));
        auto const buffer_size = ::ket::utility::integer_exp2<StateInteger>(num_buffer_qubits);
        // nonbuffer_size
        auto const nonbuffer_size
          = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment))
            / buffer_size;

        // least_permuated_nonbuffer_qubit
        auto const least_permutated_nonbuffer_qubit
          = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(num_buffer_qubits));

        // permutated_operated_buffer_qubits, permutated_operated_nonbuffer_qubits
        using std::begin;
        using std::end;
        auto const permutated_operated_buffer_qubit_first = begin(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_last = end(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_first
          = std::lower_bound(
              begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array),
              least_permutated_nonbuffer_qubit);
        auto const permutated_operated_buffer_qubit_last = permutated_operated_nonbuffer_qubit_first;
        // num_operated_buffer_qubits, num_operated_nonbuffer_qubits
# ifndef NDEBUG
        auto const num_operated_buffer_qubits
          = static_cast<BitInteger>(permutated_operated_buffer_qubit_last - permutated_operated_buffer_qubit_first);
# endif // NDEBUG
        auto const num_operated_nonbuffer_qubits
          = static_cast<BitInteger>(permutated_operated_nonbuffer_qubit_last - permutated_operated_nonbuffer_qubit_first);
        assert(num_operated_buffer_qubits + num_operated_nonbuffer_qubits == num_operated_qubits);
        // num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values
        auto const num_nonbuffer_indices_wo_operated_qubits = nonbuffer_size >> num_operated_nonbuffer_qubits;
        auto const num_operated_nonbuffer_qubit_values = ::ket::utility::integer_exp2<StateInteger>(num_operated_nonbuffer_qubits);

        // mapped_permutated_buffer_qubits
        auto const mapped_permutated_buffer_qubits
          = ::ket::mpi::inner_product_detail::generate_mapped_permutated_buffer_qubits(
              permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
              least_permutated_nonbuffer_qubit, num_operated_nonbuffer_qubits);
        auto const mapped_permutated_buffer_qubit_first = begin(mapped_permutated_buffer_qubits);
        auto const mapped_permutated_buffer_qubit_last = end(mapped_permutated_buffer_qubits);
        // num_lower_buffer_indices
        auto const num_lower_buffer_indices = mapped_permutated_buffer_qubits.empty() ? buffer_size : StateInteger{1u} << *mapped_permutated_buffer_qubit_first;

        // main loop
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        std::array<qubit_type, num_operated_qubits> modified_unsorted_qubits{
          ::ket::remove_control(permutated_qubit.qubit()),
          ::ket::remove_control(permutated_qubits.qubit())...};
        auto mapped_permutated_buffer_qubit_iter = mapped_permutated_buffer_qubit_first;
        for (auto permutated_operated_nonbuffer_qubit_iter = permutated_operated_nonbuffer_qubit_first;
             permutated_operated_nonbuffer_qubit_iter != permutated_operated_nonbuffer_qubit_last;
             ++permutated_operated_nonbuffer_qubit_iter, ++mapped_permutated_buffer_qubit_iter)
        {
          auto const found
            = std::find(
                begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
                permutated_operated_nonbuffer_qubit_iter->qubit());
          if (found != end(modified_unsorted_qubits))
            *found = mapped_permutated_buffer_qubit_iter->qubit();
        }

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        std::array<qubit_type, num_operated_qubits + 1u> modified_sorted_qubits_with_sentinel{ };
        std::copy(
          begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
          begin(modified_sorted_qubits_with_sentinel));
        modified_sorted_qubits_with_sentinel.back()
          = ::ket::make_qubit<StateInteger>(num_buffer_qubits);
        std::sort(
          begin(modified_sorted_qubits_with_sentinel),
          std::prev(end(modified_sorted_qubits_with_sentinel)));
# else // KET_USE_BIT_MASKS_EXPLICITLY
        std::array<StateInteger, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks_from_tuple(modified_unsorted_qubits, qubit_masks);
        std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks_from_tuple(modified_unsorted_qubits, index_masks);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

        auto const intercircuit_rank = intercircuit_communicator.rank(environment);

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums](
              auto const first, auto const last)
            {
              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // n0n0n
                auto const base_nonbuffer_index
                  = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                      permutated_operated_nonbuffer_qubit_first,
                      permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_aware_first_index = StateInteger{0u};
                       buffer_aware_first_index < buffer_size;
                       buffer_aware_first_index += num_lower_buffer_indices)
                  {
                    auto const nonbuffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                          buffer_aware_first_index,
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          permutated_operated_nonbuffer_qubit_first,
                          base_nonbuffer_index, num_buffer_qubits);

                    auto const buffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_index(
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          buffer_aware_first_index, mapped_buffer_qubit_bits,
                          num_buffer_qubits);

                    auto const chunk_first
                      = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                    yampi::broadcast(
                      yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices),
                      intercircuit_root, intercircuit_communicator, environment);
                  }

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_aware_first, buffer_aware_first, index_wo_qubits,
                             modified_unsorted_qubits, modified_sorted_qubits_with_sentinel);
                    });
                }
              }
            });
        else // if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             buffer_first, num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums](
              auto const first, auto const last)
            {
              auto const buffer_last = buffer_first + buffer_size;

              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_iter = buffer_first; buffer_iter != buffer_last; buffer_iter += num_lower_buffer_indices)
                    yampi::broadcast(
                      yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices),
                      intercircuit_root, intercircuit_communicator, environment);

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, buffer_first, &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_first, buffer_aware_first, index_wo_qubits,
                             modified_unsorted_qubits, modified_sorted_qubits_with_sentinel);
                    });
                }
              }
            });
# else // KET_USE_BIT_MASKS_EXPLICITLY
        if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &qubit_masks, &index_masks, &partial_sums](
              auto const first, auto const last)
            {
              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // n0n0n
                auto const base_nonbuffer_index
                  = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                      permutated_operated_nonbuffer_qubit_first,
                      permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_aware_first_index = StateInteger{0u};
                       buffer_aware_first_index < buffer_size;
                       buffer_aware_first_index += num_lower_buffer_indices)
                  {
                    auto const nonbuffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                          buffer_aware_first_index,
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          permutated_operated_nonbuffer_qubit_first,
                          base_nonbuffer_index, num_buffer_qubits);

                    auto const buffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_index(
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          buffer_aware_first_index, mapped_buffer_qubit_bits,
                          num_buffer_qubits);

                    auto const chunk_first
                      = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                    yampi::broadcast(
                      yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices),
                      intercircuit_root, intercircuit_communicator, environment);
                  }

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, &qubit_masks, &index_masks, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_aware_first, buffer_aware_first, index_wo_qubits,
                             qubit_masks, index_masks);
                    });
                }
              }
            });
        else // if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             buffer_first, num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &qubit_masks, &index_masks, &partial_sums](
              auto const first, auto const last)
            {
              auto const buffer_last = buffer_first + buffer_size;
              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_iter = buffer_first; buffer_iter != buffer_last; buffer_iter += num_lower_buffer_indices)
                    yampi::broadcast(
                      yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices),
                      intercircuit_root, intercircuit_communicator, environment);

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, buffer_first, &qubit_masks, &index_masks, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_first, buffer_aware_first, index_wo_qubits,
                             qubit_masks, index_masks);
                    });
                }
              }
            });
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator, typename DerivedDatatype, std::size_t num_operated_qubits,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl_p0(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        std::array< ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> >, num_operated_qubits > const& sorted_permutated_operated_qubits_array,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        static_assert(num_operated_qubits == sizeof...(Qubits) + 1u, "The number of permutated_qubit's is the same as num_operated_qubits");

        // nonbuffer buffer
        //   xxxxx|xxxxxxxxxx
        //    ^ ^  ^ ^   ^    <- operated qubits
        //   n n n            <- nonbuffer_index_wo_operated_qubits
        //          * *       <- mapped_buffer_qubit_bits
        //    @ @             <- operated_nonbuffer_qubit_bits
        //         u u dddddd <- "buffer_index_wo_mapped_qubits"
        //         u u        <- upper_buffer_index_wo_mapped_qubits
        //             dddddd <- "lower_buffer_index_wo_mapped_qubits"
        //    @ @  u*u*dddddd <- "buffer_aware_index" (** => @@)

        // buffer_first, buffer_last, buffer_size, num_buffer_qubits
        auto const buffer_first = ::ket::mpi::utility::buffer_begin(local_state, buffer);
        auto const num_buffer_qubits
          = ::ket::utility::integer_log2<BitInteger>(
              static_cast<StateInteger>(::ket::mpi::utility::buffer_end(local_state, buffer) - buffer_first));
        auto const buffer_size = ::ket::utility::integer_exp2<StateInteger>(num_buffer_qubits);
        // nonbuffer_size
        auto const nonbuffer_size
          = static_cast<StateInteger>(::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment))
            / buffer_size;

        // least_permuated_nonbuffer_qubit
        auto const least_permutated_nonbuffer_qubit
          = ::ket::mpi::make_permutated(::ket::make_qubit<StateInteger>(num_buffer_qubits));

        // permutated_operated_buffer_qubits, permutated_operated_nonbuffer_qubits
        using std::begin;
        using std::end;
        auto const permutated_operated_buffer_qubit_first = begin(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_last = end(sorted_permutated_operated_qubits_array);
        auto const permutated_operated_nonbuffer_qubit_first
          = std::lower_bound(
              begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array),
              least_permutated_nonbuffer_qubit);
        auto const permutated_operated_buffer_qubit_last = permutated_operated_nonbuffer_qubit_first;
        // num_operated_buffer_qubits, num_operated_nonbuffer_qubits
# ifndef NDEBUG
        auto const num_operated_buffer_qubits
          = static_cast<BitInteger>(permutated_operated_buffer_qubit_last - permutated_operated_buffer_qubit_first);
# endif // NDEBUG
        auto const num_operated_nonbuffer_qubits
          = static_cast<BitInteger>(permutated_operated_nonbuffer_qubit_last - permutated_operated_nonbuffer_qubit_first);
        assert(num_operated_buffer_qubits + num_operated_nonbuffer_qubits == num_operated_qubits);
        // num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values
        auto const num_nonbuffer_indices_wo_operated_qubits = nonbuffer_size >> num_operated_nonbuffer_qubits;
        auto const num_operated_nonbuffer_qubit_values = ::ket::utility::integer_exp2<StateInteger>(num_operated_nonbuffer_qubits);

        // mapped_permutated_buffer_qubits
        auto const mapped_permutated_buffer_qubits
          = ::ket::mpi::inner_product_detail::generate_mapped_permutated_buffer_qubits(
              permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
              least_permutated_nonbuffer_qubit, num_operated_nonbuffer_qubits);
        auto const mapped_permutated_buffer_qubit_first = begin(mapped_permutated_buffer_qubits);
        auto const mapped_permutated_buffer_qubit_last = end(mapped_permutated_buffer_qubits);
        // num_lower_buffer_indices
        auto const num_lower_buffer_indices = mapped_permutated_buffer_qubits.empty() ? buffer_size : StateInteger{1u} << *mapped_permutated_buffer_qubit_first;

        // main loop
        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        std::array<qubit_type, num_operated_qubits> modified_unsorted_qubits{
          ::ket::remove_control(permutated_qubit.qubit()),
          ::ket::remove_control(permutated_qubits.qubit())...};
        auto mapped_permutated_buffer_qubit_iter = mapped_permutated_buffer_qubit_first;
        for (auto permutated_operated_nonbuffer_qubit_iter = permutated_operated_nonbuffer_qubit_first;
             permutated_operated_nonbuffer_qubit_iter != permutated_operated_nonbuffer_qubit_last;
             ++permutated_operated_nonbuffer_qubit_iter, ++mapped_permutated_buffer_qubit_iter)
        {
          auto const found
            = std::find(
                begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
                permutated_operated_nonbuffer_qubit_iter->qubit());
          if (found != end(modified_unsorted_qubits))
            *found = mapped_permutated_buffer_qubit_iter->qubit();
        }

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        std::array<qubit_type, num_operated_qubits + 1u> modified_sorted_qubits_with_sentinel{ };
        std::copy(
          begin(modified_unsorted_qubits), end(modified_unsorted_qubits),
          begin(modified_sorted_qubits_with_sentinel));
        modified_sorted_qubits_with_sentinel.back()
          = ::ket::make_qubit<StateInteger>(num_buffer_qubits);
        std::sort(
          begin(modified_sorted_qubits_with_sentinel),
          std::prev(end(modified_sorted_qubits_with_sentinel)));
# else // KET_USE_BIT_MASKS_EXPLICITLY
        std::array<StateInteger, num_operated_qubits> qubit_masks{};
        ::ket::gate::gate_detail::make_qubit_masks_from_tuple(modified_unsorted_qubits, qubit_masks);
        std::array<StateInteger, num_operated_qubits + 1u> index_masks{};
        ::ket::gate::gate_detail::make_index_masks_from_tuple(modified_unsorted_qubits, index_masks);
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        using complex_type = ::ket::utility::meta::range_value_t<LocalState>;
        auto partial_sums = std::vector<complex_type>(::ket::utility::num_threads(parallel_policy));

        auto const intercircuit_rank = intercircuit_communicator.rank(environment);

# ifndef KET_USE_BIT_MASKS_EXPLICITLY
        if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, &datatype, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums](
              auto const first, auto const last)
            {
              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // n0n0n
                auto const base_nonbuffer_index
                  = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                      permutated_operated_nonbuffer_qubit_first,
                      permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_aware_first_index = StateInteger{0u};
                       buffer_aware_first_index < buffer_size;
                       buffer_aware_first_index += num_lower_buffer_indices)
                  {
                    auto const nonbuffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                          buffer_aware_first_index,
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          permutated_operated_nonbuffer_qubit_first,
                          base_nonbuffer_index, num_buffer_qubits);

                    auto const buffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_index(
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          buffer_aware_first_index, mapped_buffer_qubit_bits,
                          num_buffer_qubits);

                    auto const chunk_first
                      = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                    yampi::broadcast(
                      yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices, datatype),
                      intercircuit_root, intercircuit_communicator, environment);
                  }

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_aware_first, buffer_aware_first, index_wo_qubits,
                             modified_unsorted_qubits, modified_sorted_qubits_with_sentinel);
                    });
                }
              }
            });
        else // if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, &datatype, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             buffer_first, num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums](
              auto const first, auto const last)
            {
              auto const buffer_last = buffer_first + buffer_size;

              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_iter = buffer_first; buffer_iter != buffer_last; buffer_iter += num_lower_buffer_indices)
                    yampi::broadcast(
                      yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices, datatype),
                      intercircuit_root, intercircuit_communicator, environment);

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, buffer_first, &modified_unsorted_qubits, &modified_sorted_qubits_with_sentinel, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_first, buffer_aware_first, index_wo_qubits,
                             modified_unsorted_qubits, modified_sorted_qubits_with_sentinel);
                    });
                }
              }
            });
# else // KET_USE_BIT_MASKS_EXPLICITLY
        if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, &datatype, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &qubit_masks, &index_masks, &partial_sums](
              auto const first, auto const last)
            {
              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // n0n0n
                auto const base_nonbuffer_index
                  = ::ket::mpi::inner_product_detail::base_nonbuffer_index(
                      permutated_operated_nonbuffer_qubit_first,
                      permutated_operated_nonbuffer_qubit_last,
                      nonbuffer_index_wo_operated_qubits, num_buffer_qubits);

                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_aware_first_index = StateInteger{0u};
                       buffer_aware_first_index < buffer_size;
                       buffer_aware_first_index += num_lower_buffer_indices)
                  {
                    auto const nonbuffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_aware_index_to_nonbuffer_index(
                          buffer_aware_first_index,
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          permutated_operated_nonbuffer_qubit_first,
                          base_nonbuffer_index, num_buffer_qubits);

                    auto const buffer_first_index
                      = ::ket::mpi::inner_product_detail::buffer_index(
                          mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                          buffer_aware_first_index, mapped_buffer_qubit_bits,
                          num_buffer_qubits);

                    auto const chunk_first
                      = first + ((nonbuffer_first_index << num_buffer_qubits) + buffer_first_index);

                    yampi::broadcast(
                      yampi::make_buffer(chunk_first, chunk_first + num_lower_buffer_indices, datatype),
                      intercircuit_root, intercircuit_communicator, environment);
                  }

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, &qubit_masks, &index_masks, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_aware_first, buffer_aware_first, index_wo_qubits,
                             qubit_masks, index_masks);
                    });
                }
              }
            });
        else // if (intercircuit_rank == intercircuit_root)
          ::ket::mpi::utility::for_each_local_range(
            mpi_policy, local_state, circuit_communicator, environment,
            [parallel_policy, &datatype, intercircuit_root, &intercircuit_communicator, &environment, &observable,
             buffer_first, num_buffer_qubits, buffer_size,
             permutated_operated_buffer_qubit_first, permutated_operated_buffer_qubit_last,
             permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
             num_nonbuffer_indices_wo_operated_qubits, num_operated_nonbuffer_qubit_values,
             mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
             num_lower_buffer_indices,
             &qubit_masks, &index_masks, &partial_sums](
              auto const first, auto const last)
            {
              auto const buffer_last = buffer_first + buffer_size;

              // nnn
              for (auto nonbuffer_index_wo_operated_qubits = StateInteger{0u};
                   nonbuffer_index_wo_operated_qubits < num_nonbuffer_indices_wo_operated_qubits;
                   ++nonbuffer_index_wo_operated_qubits)
              {
                // **
                for (auto mapped_buffer_qubit_bits = StateInteger{0u};
                     mapped_buffer_qubit_bits < num_operated_nonbuffer_qubit_values;
                     ++mapped_buffer_qubit_bits)
                {
                  for (auto buffer_iter = buffer_first; buffer_iter != buffer_last; buffer_iter += num_lower_buffer_indices)
                    yampi::broadcast(
                      yampi::make_buffer(buffer_iter, buffer_iter + num_lower_buffer_indices, datatype),
                      intercircuit_root, intercircuit_communicator, environment);

                  auto const buffer_aware_first
                    = ::ket::mpi::inner_product_detail::make_buffer_aware_iterator(
                        first,
                        mapped_permutated_buffer_qubit_first, mapped_permutated_buffer_qubit_last,
                        permutated_operated_nonbuffer_qubit_first, permutated_operated_nonbuffer_qubit_last,
                        nonbuffer_index_wo_operated_qubits, mapped_buffer_qubit_bits, num_buffer_qubits);

                  ::ket::utility::loop_n(
                    parallel_policy, buffer_size >> num_operated_qubits,
                    [&observable, buffer_first, &qubit_masks, &index_masks, &partial_sums, buffer_aware_first](
                      StateInteger const index_wo_qubits, int const thread_index)
                    {
                      partial_sums[thread_index]
                        += observable(
                             buffer_first, buffer_aware_first, index_wo_qubits,
                             qubit_masks, index_masks);
                    });
                }
              }
            });
# endif // KET_USE_BIT_MASKS_EXPLICITLY

        return std::accumulate(begin(partial_sums), end(partial_sums), complex_type{});
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          ::ket::utility::all_in_state_vector(
            static_cast<BitInteger>(
              ::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, circuit_communicator, environment)),
            permutated_qubit.qubit(), permutated_qubits.qubit()...));

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
        constexpr auto num_operated_qubits = BitInteger{sizeof...(Qubits) + 1u};
        std::array<permutated_qubit_type, num_operated_qubits> sorted_permutated_operated_qubits_array{permutated_qubit, permutated_qubits...};
        using std::begin;
        using std::end;
        std::sort(begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array));

        if (::ket::mpi::page::is_on_page(sorted_permutated_operated_qubits_array.back(), local_state))
          return ::ket::mpi::dispatch::inner_product_page<std::remove_const_t<std::remove_reference_t<LocalState>>>::call(
            mpi_policy, parallel_policy,
            local_state, buffer, intercircuit_root, intercircuit_communicator, environment,
            sorted_permutated_operated_qubits_array,
            std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);

        return ::ket::mpi::inner_product_detail::inner_product_impl_p0(
          mpi_policy, parallel_policy,
          local_state, buffer, circuit_communicator,
          intercircuit_root, intercircuit_communicator, environment,
          sorted_permutated_operated_qubits_array,
          std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename BufferAllocator, typename DerivedDatatype,
        typename Observable, typename StateInteger, typename BitInteger, typename... Qubits>
      inline auto inner_product_impl(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        Observable&& observable,
        ::ket::mpi::permutated< ::ket::qubit<StateInteger, BitInteger> > const permutated_qubit,
        ::ket::mpi::permutated<Qubits> const... permutated_qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          ::ket::utility::all_in_state_vector(
            static_cast<BitInteger>(
              ::ket::mpi::utility::policy::num_local_qubits(mpi_policy, local_state, circuit_communicator, environment)),
            permutated_qubit.qubit(), permutated_qubits.qubit()...));

        using qubit_type = ::ket::qubit<StateInteger, BitInteger>;
        using permutated_qubit_type = ::ket::mpi::permutated<qubit_type>;
        constexpr auto num_operated_qubits = BitInteger{sizeof...(Qubits) + 1u};
        std::array<permutated_qubit_type, num_operated_qubits> sorted_permutated_operated_qubits_array{permutated_qubit, permutated_qubits...};
        using std::begin;
        using std::end;
        std::sort(begin(sorted_permutated_operated_qubits_array), end(sorted_permutated_operated_qubits_array));

        if (::ket::mpi::page::is_on_page(sorted_permutated_operated_qubits_array.back(), local_state))
          return ::ket::mpi::dispatch::inner_product_page<std::remove_const_t<std::remove_reference_t<LocalState>>>::call(
            mpi_policy, parallel_policy,
            local_state, buffer, datatype, intercircuit_root, intercircuit_communicator, environment,
            sorted_permutated_operated_qubits_array,
            std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);

        return ::ket::mpi::inner_product_detail::inner_product_impl_p0(
          mpi_policy, parallel_policy,
          local_state, buffer, datatype, circuit_communicator,
          intercircuit_root, intercircuit_communicator, environment,
          sorted_permutated_operated_qubits_array,
          std::forward<Observable>(observable), permutated_qubit, permutated_qubits...);
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        Observable&& observable,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, circuit_communicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          circuit_communicator, environment);

        return result;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype, typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        Observable&& observable,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> ::ket::utility::meta::range_value_t<LocalState>
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, circuit_communicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, datatype, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::all_reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          circuit_communicator, environment);

        return result;
      }
    } // namespace inner_product_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator,
      typename BufferAllocator, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      ::ket::utility::meta::range_value_t<LocalState> >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    // reduce version
    namespace inner_product_detail
    {
      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        Observable&& observable,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, circuit_communicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result), yampi::binary_operation{::yampi::tags::plus},
          circuit_root, circuit_communicator, environment);

        if (circuit_communicator.rank(environment) != circuit_root)
          return boost::none;

        return result;
      }

      template <
        typename MpiPolicy, typename ParallelPolicy,
        typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
        typename DerivedDatatype, typename Observable, typename... Qubits>
      inline auto inner_product(
        MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
        LocalState& local_state,
        ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
        std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
        yampi::datatype_base<DerivedDatatype> const& datatype,
        yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
        yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
        yampi::environment const& environment,
        Observable&& observable,
        ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
      -> boost::optional< ::ket::utility::meta::range_value_t<LocalState> >
      {
        assert(
          circuit_communicator.size(environment) * intercircuit_communicator.size(environment)
          == yampi::communicator{yampi::tags::world_communicator}.size(environment));

        // nonlocal operated qubits => local swap qubits
        ::ket::mpi::utility::maybe_interchange_qubits(
          mpi_policy, parallel_policy,
          local_state, permutation, buffer, datatype, circuit_communicator, environment, qubit, qubits...);

        ::ket::mpi::utility::resize_buffer_if_empty(
          local_state, buffer,
          ::ket::mpi::utility::policy::data_block_size(mpi_policy, local_state, circuit_communicator, environment));
        auto result
          = ::ket::mpi::inner_product_detail::inner_product_impl(
              mpi_policy, parallel_policy,
              local_state, buffer, datatype, circuit_communicator,
              intercircuit_root, intercircuit_communicator, environment,
              std::forward<Observable>(observable), permutation[qubit], permutation[qubits]...);

        yampi::reduce(
          yampi::in_place, yampi::make_buffer(result, datatype), yampi::binary_operation{::yampi::tags::plus},
          circuit_root, circuit_communicator, environment);

        if (circuit_communicator.rank(environment) != circuit_root)
          return boost::none;

        return result;
      }
    } // namespace inner_product_detail

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename MpiPolicy, typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::mpi::utility::policy::meta::is_mpi_policy<MpiPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      MpiPolicy const& mpi_policy, ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      ::ket::mpi::utility::log_with_time_guard<char> print{
        ::ket::mpi::gate::detail::append_qubits_string(std::string{"Inner product with observable for qubits"}, qubit, qubits...),
        environment};

      return ::ket::mpi::inner_product_detail::inner_product(
        mpi_policy, parallel_policy,
        local_state, permutation, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      (not ::ket::mpi::utility::policy::meta::is_mpi_policy<LocalState>::value)
        and (not ::ket::utility::policy::meta::is_loop_n_policy<LocalState>::value),
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(),
        ::ket::utility::policy::make_sequential(),
        local_state, permutation, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }

    template <
      typename ParallelPolicy,
      typename LocalState, typename StateInteger, typename BitInteger, typename Allocator, typename BufferAllocator,
      typename DerivedDatatype, typename Observable, typename... Qubits>
    inline std::enable_if_t<
      ::ket::utility::policy::meta::is_loop_n_policy<ParallelPolicy>::value,
      boost::optional< ::ket::utility::meta::range_value_t<LocalState> > >
    inner_product(
      ParallelPolicy const parallel_policy,
      LocalState& local_state,
      ::ket::mpi::qubit_permutation<StateInteger, BitInteger, Allocator>& permutation,
      std::vector< ::ket::utility::meta::range_value_t<LocalState>, BufferAllocator >& buffer,
      yampi::datatype_base<DerivedDatatype> const& datatype,
      yampi::rank const circuit_root, yampi::communicator const& circuit_communicator,
      yampi::rank const intercircuit_root, yampi::communicator const& intercircuit_communicator,
      yampi::environment const& environment,
      Observable&& observable, ::ket::qubit<StateInteger, BitInteger> const qubit, Qubits const... qubits)
    {
      return ::ket::mpi::inner_product(
        ::ket::mpi::utility::policy::make_simple_mpi(), parallel_policy,
        local_state, permutation, buffer, datatype, circuit_root, circuit_communicator,
        intercircuit_root, intercircuit_communicator, environment,
        std::forward<Observable>(observable), qubit, qubits...);
    }
  } // namespace mpi
} // namespace ket


#endif // KET_MPI_INNER_PRODUCT_HPP
