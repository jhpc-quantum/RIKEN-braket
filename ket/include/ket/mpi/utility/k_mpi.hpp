#ifndef KET_MPI_UTILITY_K_MPI_HPP
#define KET_MPI_UTILITY_K_MPI_HPP

# ifdef __FUJITSU
#   include <cstddef>
#   ifndef KET_USE_CPP03
#     include <cstdint>
#   else
#     include <boost/cstdint.hpp>
#   endif
#   include <cassert>
#   include <vector>
#   include <iterator>
#   include <utility>
#   include <stdexcept>
#   ifndef KET_USE_CPP03
#     include <array>
#     ifndef NDEBUG
#       include <type_traits>
#     endif
#   else
#     include <boost/array.hpp>
#     include <boost/utility.hpp>
#     include <boost/static_assert.hpp>
#     include <boost/range/begin.hpp>
#     include <boost/range/end.hpp>
#     ifndef NDEBUG
#       include <boost/type_traits/is_unsigned.hpp>
#     endif
#   endif

#   include <boost/range/size.hpp>
#   include <boost/range/empty.hpp>
#   include <boost/range/algorithm/find.hpp>
#   include <boost/range/algorithm/find_if.hpp>

#   include <yampi/communicator.hpp>
#   include <yampi/request.hpp>
#   include <yampi/send_receive.hpp>

#   include <mpi-ext.h>

#   include <ket/qubit.hpp>
#   include <ket/utility/integer_exp2.hpp>
#   include <ket/utility/integer_log2.hpp>
#   include <ket/mpi/qubit_permutation.hpp>
#   include <ket/mpi/utility/general_mpi.hpp>
#   include <ket/mpi/utility/nonblocking_send_receive.hpp>
#   include <ket/mpi/utility/detail/swap_local_qubits.hpp>

#   ifndef KET_USE_CPP03
#     define KET_uint64_t std::uint64_t
#     define KET_is_unsigned std::is_unsigned
#     define KET_array std::array
#     define KET_begin std::begin
#     define KET_end std::end
#     define KET_next std::next
#   else
#     define KET_uint64_t boost::uint64_t
#     define KET_is_unsigned boost::is_unsigned
#     define KET_array boost::array
#     define KET_begin boost::begin
#     define KET_end boost::end
#     define KET_next boost::next
#     define static_assert(exp, msg) BOOST_STATIC_ASSERT_MSG(exp, msg)
#     define noexcept throw()
#     define constexpr 
#   endif


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      class invalid_k_mpi_initialization_error
        : public std::logic_error
      {
#   ifndef KET_USE_CPP03
        using super_type = std::logic_error;
#   else
        typedef std::logic_error super_type;
#   endif

       public:
#   ifndef KET_USE_CPP03
        invalid_k_mpi_initialization_error(char const* argument)
          : super_type{argument}
        { }

        invalid_k_mpi_initialization_error(std::string const& argument)
          : super_type{argument}
        { }
#   else
        invalid_k_mpi_initialization_error(char const* argument)
          : super_type(argument)
        { }

        invalid_k_mpi_initialization_error(std::string const& argument)
          : super_type(argument)
        { }
#   endif
      };

      namespace k_mpi_detail
      {
#   ifdef KET_USE_CPP03
        template <typename Value>
        struct is_dividable
        {
          Value value_;

          is_dividable(Value const value) : value_(value) { }

          bool operator()(Value const v) const
          { return v % value_ == static_cast<Value>(0); }
        };
#   endif

        template <typename StateInteger = KET_uint64_t>
        class coordinate
        {
#   ifndef KET_USE_CPP03
          StateInteger group_ = 0u;
          StateInteger unit_ = 0u;
          StateInteger node_ = 0u;
#   else
          StateInteger group_;
          StateInteger unit_;
          StateInteger node_;
#   endif

         public:
#   ifndef KET_USE_CPP03
          coordinate() = default;
          constexpr coordinate(
            StateInteger const group,
            StateInteger const unit, StateInteger const node)
            : group_{group}, unit_{unit}, node_{node}
          { }
#   else
          coordinate()
            : group_(0u), unit_(0u), node_(0u)
          { }

          coordinate(
            StateInteger const group,
            StateInteger const unit, StateInteger const node)
            : group_(group), unit_(unit), node_(node)
          { }
#   endif

          constexpr StateInteger group() const { return group_; }
          constexpr StateInteger unit() const { return unit_; }
          constexpr StateInteger node() const { return node_; }
        };

        template <typename StateInteger = KET_uint64_t>
        class xindices
        {
#   ifndef KET_USE_CPP03
          StateInteger uindex_ = 0u;
          StateInteger nindex_ = 0u;
          StateInteger lindex_ = 0u;
#   else
          StateInteger uindex_;
          StateInteger nindex_;
          StateInteger lindex_;
#   endif

         public:
#   ifndef KET_USE_CPP03
          xindices() = default;
          constexpr xindices(
            StateInteger const uindex,
            StateInteger const nindex, StateInteger const lindex)
            : uindex_{uindex}, nindex_{nindex}, lindex_{lindex}
          { }
#   else
          xindices()
            : uindex_(0u), nindex_(0u), lindex_(0u)
          { }

          constexpr xindices(
            StateInteger const uindex,
            StateInteger const nindex, StateInteger const lindex)
            : uindex_(uindex), nindex_(nindex), lindex_(lindex)
          { }
#   endif

          constexpr StateInteger uindex() const { return uindex_; }
          constexpr StateInteger nindex() const { return nindex_; }
          constexpr StateInteger lindex() const { return lindex_; }
        };

        struct const_nums
        {
#   ifndef KET_USE_CPP03
          static constexpr std::size_t num_node_unit_qubits = 3u;
          static constexpr std::size_t center_unit = 4u;
          static constexpr std::size_t center_node = 4u;
#   else
          static std::size_t const num_node_unit_qubits = 3u;
          static std::size_t const center_unit = 4u;
          static std::size_t const center_node = 4u;
#   endif
        };
      }

      namespace policy
      {
        template <
          std::size_t num_condition_qubits_ = 6u, std::size_t threshold_ = 7u,
          typename StateInteger = KET_uint64_t>
        class k_mpi
        {
          static bool is_initialized_;

          static int mpi_size_;
          static yampi::rank mpi_rank_;

          static std::size_t num_groups_;

          static std::size_t num_qubits_;
          static std::size_t num_gqubits_;
          static std::size_t num_lqubits_;

          // system_*: system coordinate, my_*: coordinate in this code
          static int system_size_[3u];
          static std::size_t system_to_my_[3u];
          static std::size_t my_to_system_[3u];
          static int my_group_size_[3u];

          static ket::mpi::utility::k_mpi_detail::coordinate<StateInteger>
            coordinate_;
          static std::size_t local_state_size_;

         public:
          k_mpi(std::size_t const num_qubits, yampi::communicator const communicator)
          {
            if (is_initialized_)
              return;

            mpi_size_ = communicator.size();
            mpi_rank_ = communicator.rank();

            num_groups_ = mpi_size_ / 81u;
            if (mpi_size_ % 81u != 0u
                or ket::utility::integer_exp2<std::size_t>(
                     ket::utility::integer_log2<std::size_t>(num_groups_))
                   != num_groups_)
#   ifndef KET_USE_CPP03
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "wrong number of MPI processes created"};
#   else
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "wrong number of MPI processes created");
#   endif

            num_qubits_ = num_qubits;
            num_gqubits_ = ket::utility::integer_log2<std::size_t>(num_groups_);

            if (num_qubits <= (num_gqubits_+2u*num_condition_qubits_+6u))
#   ifndef KET_USE_CPP03
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "wrong number of MPI processes created"};
#   else
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "wrong number of MPI processes created");
#   endif
            num_lqubits_
              = num_qubits-(num_gqubits_+2u*num_condition_qubits_+6u);

            int dimension;
#   ifndef KET_USE_CPP03
            auto status = FJMPI_Topology_get_dimension(&dimension);
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "called from dynamically created MPI process"};
            if (dimension != 3)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "dimension of MPI processes should be 3"};
#   else
            int status = FJMPI_Topology_get_dimension(&dimension);
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "called from dynamically created MPI process");
            if (dimension != 3)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "dimension of MPI processes should be 3");
#   endif

            status
              = FJMPI_Topology_get_shape(
                  &system_size_[0u], &system_size_[1u], &system_size_[2u]);
#   ifndef KET_USE_CPP03
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "called from dynamically created MPI process"};
#   else
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "called from dynamically created MPI process");
#   endif

            using KET_end;
#   ifndef KET_USE_CPP03
            auto const my_x_iter
              = boost::find_if(
                  system_size_,
                  [](int const value) { return value % 27 == 0; });
            if (my_x_iter == end(system_size_))
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "group size should be 27x3x1 (or its permutation)"};
#   else
            int* const my_x_iter
              = boost::find_if(
                  system_size_,
                  ket::mpi::utility::k_mpi_detail::is_dividable<int>(27));
            if (my_x_iter == end(system_size_))
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "group size should be 27x3x1 (or its permutation)");
#   endif
            using KET_begin;
            my_to_system_[0u] = my_x_iter-begin(system_size_);

            using KET_next;
#   ifndef KET_USE_CPP03
            auto const possible_my_y_iter1
              = std::find_if(
                  begin(system_size_), my_x_iter,
                  [] (int const value) { return value % 3 == 0 });
            auto const possible_my_y_iter2
              = std::find_if(
                  next(my_x_iter), end(system_size_),
                  [] (int const value) { return value % 3 == 0 });
            auto const my_y_iter
              = possible_my_y_iter1 == my_x_iter
                ? possible_my_y_iter2
                : possible_my_y_iter1;
            if (my_y_iter == end(system_size_))
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "group size should be 27x3x1 (or its permutation)"};
#   else
            int* const possible_my_y_iter1
              = std::find_if(
                  begin(system_size_), my_x_iter,
                  ket::mpi::utility::k_mpi_detail::is_dividable<int>(3));
            int* const possible_my_y_iter2
              = std::find_if(
                  next(my_x_iter), end(system_size_),
                  ket::mpi::utility::k_mpi_detail::is_dividable<int>(3));
            int* const my_y_iter
              = possible_my_y_iter1 == my_x_iter
                ? possible_my_y_iter2
                : possible_my_y_iter1;
            if (my_y_iter == end(system_size_))
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "group size should be 27x3x1 (or its permutation)");
#   endif
            my_to_system_[1u] = my_y_iter-begin(system_size_);

            my_to_system_[2u] = 3 - my_to_system_[0u] - my_to_system_[1u];

            my_group_size_[0u] = system_size_[my_to_system_[0u]] / 27u;
            my_group_size_[1u] = system_size_[my_to_system_[1u]] / 3u;
            my_group_size_[2u] = system_size_[my_to_system_[2u]];

            system_to_my_[my_to_system_[0u]] = 0u;
            system_to_my_[my_to_system_[1u]] = 1u;
            system_to_my_[my_to_system_[2u]] = 2u;


            coordinate_ = rank_to_coordinate(mpi_rank_);
            local_state_size_ = local_state_size(mpi_rank_);


            is_initialized_ = true;
          }

          static int mpi_size()
          { assert(is_initialized_); return mpi_size_; }
          static yampi::rank mpi_rank()
          { assert(is_initialized_); return mpi_rank_; }

          static std::size_t num_groups()
          { assert(is_initialized_); return num_groups_; }

          static std::size_t num_qubits()
          { assert(is_initialized_); return num_qubits_; }
          static std::size_t num_gqubits()
          { assert(is_initialized_); return num_gqubits_; }
          static std::size_t num_uqubits()
          { assert(is_initialized_); return num_node_unit_qubits(); }
          static std::size_t num_ucqubits()
          { assert(is_initialized_); return num_condition_qubits(); }
          static std::size_t num_nqubits()
          { assert(is_initialized_); return num_node_unit_qubits(); }
          static std::size_t num_ncqubits()
          { assert(is_initialized_); return num_condition_qubits(); }
          static std::size_t num_lqubits()
          { assert(is_initialized_); return num_lqubits_; }

          static constexpr std::size_t num_node_unit_qubits()
          {
            assert(is_initialized_);
            return ket::mpi::utility::k_mpi_detail::const_nums::num_node_unit_qubits;
          }
          static constexpr std::size_t num_condition_qubits()
          { assert(is_initialized_); return num_condition_qubits_; }

          static constexpr std::size_t center_unit()
          {
            assert(is_initialized_);
            return ket::mpi::utility::k_mpi_detail::const_nums::center_unit;
          }
          static constexpr std::size_t center_node()
          {
            assert(is_initialized_);
            return ket::mpi::utility::k_mpi_detail::const_nums::center_node;
          }

          static constexpr StateInteger threshold()
          {
            assert(is_initialized_);
#   ifndef KET_USE_CPP03
            return {threshold_};
#   else
            return static_cast<StateInteger>(threshold_);
#   endif
          }

          static constexpr
          ket::mpi::utility::k_mpi_detail::coordinate<StateInteger>
          coordinate()
          { assert(is_initialized_); return coordinate_; }

          static std::size_t local_state_size()
          { assert(is_initialized_); return local_state_size_; }


#   ifndef KET_USE_CPP03
          static std::size_t local_state_size(yampi::rank const rank)
          {
            auto const coordinate = rank_to_coordinate(rank);

            constexpr auto condition_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_condition_qubits_);
            constexpr auto node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  ket::mpi::utility::k_mpi_detail::const_nums::num_node_unit_qubits);
            
            auto const nindex_last
              = coordinate.node() == ket::mpi::utility::k_mpi_detail::const_nums::center_node
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold_)
                : condition_value_last-threshold_;
            auto const uindex_last
              = coordinate.unit() == ket::mpi::utility::k_mpi_detail::const_nums::center_unit
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold_)
                : condition_value_last-threshold_;

            return num_groups_*uindex_last*nindex_last;
          }
#   else
          static std::size_t local_state_size(yampi::rank const rank)
          {
            typedef
              ket::mpi::utility::k_mpi_detail::coordinate<StateInteger>
              coordinate_type;
            coordinate_type const coordinate = rank_to_coordinate(rank);

            StateInteger const condition_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_condition_qubits_);
            StateInteger const node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  ket::mpi::utility::k_mpi_detail::const_nums::num_node_unit_qubits);
            
            StateInteger const nindex_last
              = coordinate.node() == ket::mpi::utility::k_mpi_detail::const_nums::center_node
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold_)
                : condition_value_last-threshold_;
            StateInteger const uindex_last
              = coordinate.unit() == ket::mpi::utility::k_mpi_detail::const_nums::center_unit
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold_)
                : condition_value_last-threshold_;

            return num_groups_*uindex_last*nindex_last;
          }
#   endif


          static constexpr
          ket::mpi::utility::k_mpi_detail::coordinate<StateInteger>
          rank_to_coordinate(yampi::rank const rank)
          {
            assert(is_initialized_);

            int system_xyz[3u];
#   ifndef KET_USE_CPP03
            auto status
              = FJMPI_Topology_rank2xyz(
                  rank.mpi_rank(), &system_xyz[0u], &system_xyz[1u], &system_xyz[2u]);
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "called from dynamically created MPI process"};

            auto my_xyz[3u]
              = {system_xyz[my_to_system_[0u]], system_xyz[my_to_system_[1u]],
                 system_xyz[my_to_system_[2u]]};

            auto const group
              = static_cast<StateInteger>(
                  (my_xyz[0u]/27u) * (my_xyz[1u]/3u) * my_xyz[2u]);
            auto const unit
              = static_cast<StateInteger>((my_xyz[0u]%27u) / 3u);
            auto const node
              = static_cast<StateInteger>(
                  (my_xyz[0u]%27u)%3u + (my_xyz[1u]%3u)*3u);

            return {group, unit, node};
#   else
            int status
              = FJMPI_Topology_rank2xyz(
                  rank.mpi_rank(), &system_xyz[0u], &system_xyz[1u], &system_xyz[2u]);
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "called from dynamically created MPI process");

            int my_xyz[3u]
              = {system_xyz[my_to_system_[0u]], system_xyz[my_to_system_[1u]],
                 system_xyz[my_to_system_[2u]]};

            StateInteger const group
              = (my_xyz[0u]/27u) * (my_xyz[1u]/3u) * my_xyz[2u];
            StateInteger const unit = (my_xyz[0u]%27u) / 3u;
            StateInteger const node
              = (my_xyz[0u]%27u)%3u + (my_xyz[1u]%3u)*3u;

            return
              ket::mpi::utility::k_mpi_detail::coordinate<StateInteger>(
                group, unit, node);
#   endif
          }

          static yampi::rank coordinate_to_rank(
            StateInteger const group,
            StateInteger const unit, StateInteger const node)
          {
#   ifndef KET_USE_CPP03
            auto my_xy_in_group[2u]
              = {3*static_cast<int>(unit)+static_cast<int>(node)%3,
                 static_cast<int>(node)/3};
            auto my_group_xyz[3u]
              = {static_cast<int>(group)%my_group_size_[0u],
                 (static_cast<int>(group)/my_group_size_[0u])
                   % my_group_size_[1u],
                 (static_cast<int>(group)/my_group_size_[0u])
                   / my_group_size_[1u]};
            auto my_xyz[3u]
              = {my_group_xyz[0u]*27+my_xy_in_group[0u],
                 my_group_xyz[1u]*3+my_xy_in_group[1u],
                 my_group_xyz[2u]};
            auto system_xyz[3u]
              = {my_xyz[system_to_my_[0u]], my_xyz[system_to_my_[1u]],
                 my_xyz[system_to_my_[2u]]};

            int result;
            auto status
              = FJMPI_Topology_xyz2rank(
                  system_xyz[0u], system_xyz[1u], system_xyz[2u], &result);
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error{
                "called from dynamically created MPI process"};

            return yampi::rank{result};
#   else
            int my_xy_in_group[2u]
              = {3*static_cast<int>(unit)+static_cast<int>(node)%3,
                 static_cast<int>(node)/3};
            int my_group_xyz[3u]
              = {static_cast<int>(group)%my_group_size_[0u],
                 (static_cast<int>(group)/my_group_size_[0u])
                   % my_group_size_[1u],
                 (static_cast<int>(group)/my_group_size_[0u])
                   / my_group_size_[1u]};
            int my_xyz[3u]
              = {my_group_xyz[0u]*27+my_xy_in_group[0u],
                 my_group_xyz[1u]*3+my_xy_in_group[1u],
                 my_group_xyz[2u]};
            int system_xyz[3u]
              = {my_xyz[system_to_my_[0u]], my_xyz[system_to_my_[1u]],
                 my_xyz[system_to_my_[2u]]};

            int result;
            int status
              = FJMPI_Topology_xyz2rank(
                  system_xyz[0u], system_xyz[1u], system_xyz[2u], &result);
            if (status != 0)
              throw ket::mpi::utility::invalid_k_mpi_initialization_error(
                "called from dynamically created MPI process");

            return yampi::rank(result);
#   endif
          }

          static yampi::rank coordinate_to_rank(
            ket::mpi::utility::k_mpi_detail::coordinate<
              StateInteger> const& coord)
          { return coordinate_to_rank(coord.group(), coord.unit(), coord.node()); }

          static StateInteger xindices_to_index(
            StateInteger const uindex,
            StateInteger const nindex, StateInteger const lindex,
            StateInteger const node_condition_value)
          {
#   ifndef KET_USE_CPP03
            return node_condition_value < StateInteger{threshold_}
              ? static_cast<StateInteger>(8u*threshold_)
                  * ket::utility::integer_exp2<StateInteger>(num_lqubits_) * uindex
                + ket::utility::integer_exp2<StateInteger>(num_lqubits_) * nindex
                + lindex
              : (ket::utility::integer_exp2<StateInteger>(num_condition_qubits_)
                 - StateInteger{threshold_})
                  * ket::utility::integer_exp2<StateInteger>(num_lqubits_) * uindex
                + ket::utility::integer_exp2<StateInteger>(num_lqubits_) * nindex
                + lindex;
#   else
            return node_condition_value < static_cast<StateInteger>(threshold_)
              ? static_cast<StateInteger>(8u*threshold_)
                  * ket::utility::integer_exp2<StateInteger>(num_lqubits_) * uindex
                + ket::utility::integer_exp2<StateInteger>(num_lqubits_) * nindex
                + lindex
              : (ket::utility::integer_exp2<StateInteger>(num_condition_qubits_)
                 - static_cast<StateInteger>(threshold_))
                  * ket::utility::integer_exp2<StateInteger>(num_lqubits_) * uindex
                + ket::utility::integer_exp2<StateInteger>(num_lqubits_) * nindex
                + lindex;
#   endif
          }

          static StateInteger xindices_to_index(
            ket::mpi::utility::k_mpi_detail::xindices<
              StateInteger> const& xind,
            StateInteger const node_condition_value)
          {
            return xindices_to_index(
              xind.uindex(), xind.nindex(), xind.lindex(),
              node_condition_value);
          }

          static ket::mpi::utility::k_mpi_detail::xindices<StateInteger>
          index_to_xindices(
            StateInteger const index,
            ket::mpi::utility::k_mpi_detail::coordinate<
              StateInteger> const& coord)
          {
#   ifndef KET_USE_CPP03
            if (coord.node() == center_node())
              return {
                (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                  / (8u*threshold_),
                (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                  % (8u*threshold_),
                index % ket::utility::integer_exp2<StateInteger>(num_lqubits_)};
            return {
              (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                / (ket::utility::integer_exp2<StateInteger>(num_condition_qubits_)
                   - StateInteger{threshold_}),
              (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                % (ket::utility::integer_exp2<StateInteger>(num_condition_qubits_)
                   - StateInteger{threshold_}),
              index % ket::utility::integer_exp2<StateInteger>(num_lqubits_)};
#   else
            if (coord.node() == center_node())
              return
                ket::mpi::utility::k_mpi_detail::xindices<StateInteger>(
                  (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                    / (8u*threshold_),
                  (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                    % (8u*threshold_),
                  index % ket::utility::integer_exp2<StateInteger>(num_lqubits_));
            return
              ket::mpi::utility::k_mpi_detail::xindices<StateInteger>(
                (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                  / (ket::utility::integer_exp2<StateInteger>(num_condition_qubits_)
                     - static_cast<StateInteger>(threshold_)),
                (index / ket::utility::integer_exp2<StateInteger>(num_lqubits_))
                  % (ket::utility::integer_exp2<StateInteger>(num_condition_qubits_)
                     - static_cast<StateInteger>(threshold_)),
                index % ket::utility::integer_exp2<StateInteger>(num_lqubits_));
#   endif
          }


          static int system_size(std::size_t const n)
          { assert(is_initialized_ and n < 3u); return system_size_[n]; }
          static int system_x_size()
          { assert(is_initialized_); return system_size_[0u]; }
          static int system_y_size()
          { assert(is_initialized_); return system_size_[1u]; }
          static int system_z_size()
          { assert(is_initialized_); return system_size_[2u]; }

          static int size(std::size_t const n)
          {
            assert(is_initialized_ and n < 3u);
            return system_size_[my_to_system_[n]];
          }
          static int x_size()
          { assert(is_initialized_); return system_size_[my_to_system_[0u]]; }
          static int y_size()
          { assert(is_initialized_); return system_size_[my_to_system_[1u]]; }
          static int z_size()
          { assert(is_initialized_); return system_size_[my_to_system_[2u]]; }
        };

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        bool k_mpi<num_condition_qubits, threshold, StateInteger>
          ::is_initialized_ = false;

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        int k_mpi<num_condition_qubits, threshold, StateInteger>
          ::mpi_size_;
        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        int k_mpi<num_condition_qubits, threshold, StateInteger>
          ::mpi_rank_;

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        std::size_t k_mpi<num_condition_qubits, threshold, StateInteger>
          ::num_groups_;

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        std::size_t k_mpi<num_condition_qubits, threshold, StateInteger>
          ::num_qubits_;
        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        std::size_t k_mpi<num_condition_qubits, threshold, StateInteger>
          ::num_gqubits_;
        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        std::size_t k_mpi<num_condition_qubits, threshold, StateInteger>
          ::num_lqubits_;

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        int k_mpi<num_condition_qubits, threshold, StateInteger>
          ::system_size_[3u];
        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        std::size_t k_mpi<num_condition_qubits, threshold, StateInteger>
          ::system_to_my_[3u];
        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        std::size_t k_mpi<num_condition_qubits, threshold, StateInteger>
          ::my_to_system_[3u];
        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        int k_mpi<num_condition_qubits, threshold, StateInteger>
          ::my_group_size_[3u];

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        ket::mpi::utility::k_mpi_detail::coordinate<StateInteger>
        k_mpi<num_condition_qubits, threshold, StateInteger>
          ::coordinate_;
        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        std::size_t k_mpi<num_condition_qubits, threshold, StateInteger>
          ::local_state_size_;
          static std::size_t local_state_size_;


        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        inline constexpr
        ket::mpi::utility::policy::k_mpi<
          num_condition_qubits, threshold, StateInteger>
        make_k_mpi(
          std::size_t const num_qubits,
          yampi::communicator const communicator)
        {
#   ifndef KET_USE_CPP03
          return {num_qubits, communicator};
#   else
          typedef
            ket::mpi::utility::policy::k_mpi<
              num_condition_qubits, threshold, StateInteger>
            k_mpi;
          return k_mpi(num_qubits, communicator);
#   endif
        }
      }

      namespace k_mpi_detail
      {
#   ifndef KET_USE_CPP03
        enum class which_qubit
        { gqubit, uqubit, ucqubit, nqubit, ncqubit, lqubit };
#   else
        enum which_qubit
        { gqubit, uqubit, ucqubit, nqubit, ncqubit, lqubit };
#   endif

        inline constexpr bool is_g_operation(
          ket::mpi::utility::k_mpi_detail::which_qubit const
            operation_for_which_qubit)
        {
#   ifndef KET_USE_CPP03
          using my_which_qubit = ket::mpi::utility::k_mpi_detail::which_qubit;
          return operation_for_which_qubit == my_which_qubit::gqubit;
#   else
          return operation_for_which_qubit == gqubit;
#   endif
        }

        inline constexpr bool is_u_uc_operation(
          ket::mpi::utility::k_mpi_detail::which_qubit const
            operation_for_which_qubit)
        {
#   ifndef KET_USE_CPP03
          using my_which_qubit = ket::mpi::utility::k_mpi_detail::which_qubit;
          return
            operation_for_which_qubit == my_which_qubit::uqubit
            or operation_for_which_qubit == my_which_qubit::ucqubit;
#   else
          return
            operation_for_which_qubit == uqubit
            or operation_for_which_qubit == ucqubit;
#   endif
        }

        inline constexpr bool is_n_nc_operation(
          ket::mpi::utility::k_mpi_detail::which_qubit const
            operation_for_which_qubit)
        {
#   ifndef KET_USE_CPP03
          using my_which_qubit = ket::mpi::utility::k_mpi_detail::which_qubit;
          return
            operation_for_which_qubit == my_which_qubit::nqubit
            or operation_for_which_qubit == my_which_qubit::ncqubit;
#   else
          return
            operation_for_which_qubit == nqubit
            or operation_for_which_qubit == ncqubit;
#   endif
        }

        inline constexpr bool is_l_operation(
          ket::mpi::utility::k_mpi_detail::which_qubit const
            operation_for_which_qubit)
        {
#   ifndef KET_USE_CPP03
          using my_which_qubit = ket::mpi::utility::k_mpi_detail::which_qubit;
          return operation_for_which_qubit == my_which_qubit::lqubit;
#   else
          return operation_for_which_qubit == lqubit;
#   endif
        }


        inline constexpr bool is_node_unit_operation(
          ket::mpi::utility::k_mpi_detail::which_qubit const
            operation_for_which_qubit)
        {
#   ifndef KET_USE_CPP03
          using my_which_qubit = ket::mpi::utility::k_mpi_detail::which_qubit;
          return
            operation_for_which_qubit == my_which_qubit::uqubit
            or operation_for_which_qubit == my_which_qubit::nqubit;
#   else
          return
            operation_for_which_qubit == uqubit
            or operation_for_which_qubit == nqubit;
#   endif
        }

        inline constexpr bool is_condition_operation(
          ket::mpi::utility::k_mpi_detail::which_qubit const
            operation_for_which_qubit)
        {
#   ifndef KET_USE_CPP03
          using my_which_qubit = ket::mpi::utility::k_mpi_detail::which_qubit;
          return
            operation_for_which_qubit == my_which_qubit::ucqubit
            or operation_for_which_qubit == my_which_qubit::ncqubit;
#   else
          return
            operation_for_which_qubit == ucqubit
            or operation_for_which_qubit == ncqubit;
#   endif
        }


        template <typename StateInteger>
        inline constexpr bool is_in_center_node(
          ket::mpi::utility::k_mpi_detail::coordinate<
            StateInteger> const& coord)
        {
          return coord.node()
            == ket::mpi::utility::k_mpi_detail::const_nums::center_node;
        }

        template <typename StateInteger>
        inline constexpr bool is_in_center_unit(
          ket::mpi::utility::k_mpi_detail::coordinate<
            StateInteger> const& coord)
        {
          return coord.unit()
            == ket::mpi::utility::k_mpi_detail::const_nums::center_unit;
        }
      }

      namespace dispatch
      {
        template <std::size_t num_qubits_of_operation, typename MpiPolicy>
        struct maybe_interchange_qubits;

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        struct maybe_interchange_qubits<
          1u,
          ket::mpi::utility::policy::k_mpi<
            num_condition_qubits, threshold, StateInteger> >
        {
          template <
            typename ParallelPolicy, typename LocalState,
            typename BitInteger,
            std::size_t num_qubits_of_operation,
            std::size_t num_unswappable_qubits, typename Allocator>
          static void call(
            ParallelPolicy const parallel_policy,
            LocalState& local_state,
            KET_array<
              ket::qubit<StateInteger, BitInteger>,
              num_qubits_of_operation> const& qubits,
            KET_array<
              ket::qubit<StateInteger, BitInteger>,
              num_unswappable_qubits> const& unswappable_qubits,
            ket::mpi::qubit_permutation<
              StateInteger, BitInteger, Allocator>&
              permutation,
            yampi::communicator const communicator)
          {
            static_assert(
              KET_is_unsigned<StateInteger>::value,
              "StateInteger should be unsigned");
            static_assert(
              KET_is_unsigned<BitInteger>::value,
              "BitInteger should be unsigned");

            assert(communicator.size() > 1);


            // determine local&global swap qubits
#   ifndef KET_USE_CPP03
            auto const permutated_global_swap_qubit = permutation[qubits[0u]];
#   else
            typedef ket::qubit<StateInteger, BitInteger> qubit_type;
            qubit_type const permutated_global_swap_qubit
              = permutation[qubits[0u]];
#   endif

#   ifndef KET_USE_CPP03
            using my_k_mpi_policy = ket::mpi::utility::policy::k_mpi<
              num_condition_qubits, threshold, StateInteger>;
            auto const num_lqubits
              = static_cast<BitInteger>(my_k_mpi_policy::num_lqubits());
            constexpr auto num_node_unit_qubits
              = static_cast<BitInteger>(
                  my_k_mpi_policy::num_node_unit_qubits());
#   else
            typedef
              ket::mpi::utility::policy::k_mpi<
                num_condition_qubits, threshold, StateInteger>
              my_k_mpi_policy;
            BitInteger const num_lqubits = my_k_mpi_policy::num_lqubits();
            BitInteger const num_node_unit_qubits
              = my_k_mpi_policy::num_node_unit_qubits();
#   endif

            namespace my_k_mpi_detail = ket::mpi::utility::k_mpi_detail;
#   ifndef KET_USE_CPP03
            using my_which_qubit = my_k_mpi_detail::which_qubit;
#   else
            namespace my_which_qubit = my_k_mpi_detail;
#   endif

#   ifndef KET_USE_CPP03
            auto const operation_for_which_qubit
#   else
            my_k_mpi_detail::which_qubit const operation_for_which_qubit
#   endif
              = static_cast<BitInteger>(permutated_global_swap_qubit)
                  < num_lqubits
                ? my_which_qubit::lqubit
                : static_cast<BitInteger>(permutated_global_swap_qubit)
                    < num_lqubits+num_condition_qubits
                  ? my_which_qubit::ncqubit
                  : static_cast<BitInteger>(permutated_global_swap_qubit)
                      < num_lqubits+num_condition_qubits
                        + num_node_unit_qubits
                    ? my_which_qubit::nqubit
                    : static_cast<BitInteger>(permutated_global_swap_qubit)
                        < num_lqubits+2u*num_condition_qubits
                          + num_node_unit_qubits
                      ? my_which_qubit::ucqubit
                      : static_cast<BitInteger>(permutated_global_swap_qubit)
                          < num_lqubits+2u*num_condition_qubits
                            + 2u*num_node_unit_qubits
                        ? my_which_qubit::uqubit
                        : my_which_qubit::gqubit;

            // (1) 0 <= b < l (operation for lqubits): no-swap
            if (operation_for_which_qubit == my_which_qubit::lqubit)
              return;

#   ifndef KET_USE_CPP03
            using qubit_type = ket::qubit<StateInteger, BitInteger>;
            auto const permutated_local_swap_qubit
              = qubit_type{num_lqubits-1u};

            using ket::mpi::inverse;
            auto const local_swap_qubit
              = inverse(permutation)[permutated_local_swap_qubit];
#   else
            qubit_type const permutated_local_swap_qubit(num_lqubits-1u);

            using ket::mpi::inverse;
            qubit_type const local_swap_qubit
              = inverse(permutation)[permutated_local_swap_qubit];
#   endif

            if (not boost::empty(boost::find<boost::return_found_end>(
                      unswappable_qubits, local_swap_qubit)))
            {
#   ifndef KET_USE_CPP03
              auto permutated_other_qubit = qubit_type{num_lqubits-1u};
              auto other_qubit = qubit_type{};
#   else
              qubit_type permutated_other_qubit(num_lqubits-1u);
              qubit_type other_qubit;
#   endif
              do
              {
                --permutated_other_qubit;
                using ket::mpi::inverse;
                other_qubit
                  = inverse(permutation)[permutated_other_qubit];
              }
              while (not boost::empty(boost::find<boost::return_found_end>(
                           unswappable_qubits, other_qubit)));

              using KET_begin;
              using KET_end;
              ket::mpi::utility::detail::swap_local_qubits(
                parallel_policy, local_state,
                permutated_local_swap_qubit, permutated_other_qubit);
              using ket::mpi::permutate;
              permutate(permutation, local_swap_qubit, other_qubit);
            }


#   ifndef KET_USE_CPP03
            // ...|001000|...|0000000000
            auto const permutated_global_swap_qubit_mask
              = StateInteger{1u} << permutated_global_swap_qubit;
            // ...|000000|...|1000000000
            auto const permutated_local_swap_qubit_mask
              = StateInteger{1u} << permutated_local_swap_qubit;
#   else
            // ...|001000|...|0000000000
            StateInteger const permutated_global_swap_qubit_mask
              = static_cast<StateInteger>(1u) << permutated_global_swap_qubit;
            // ...|000000|...|1000000000
            StateInteger const permutated_local_swap_qubit_mask
              = static_cast<StateInteger>(1u) << permutated_local_swap_qubit;
#   endif

#   ifndef KET_USE_CPP03
            auto const rank = communicator.rank();
            // coordinate := (group, unit, node)
            auto const coordinate
              = my_k_mpi_policy::rank_to_coordinate(rank);
#   else
            yampi::rank const rank = communicator.rank();
            // coordinate := (group, unit, node)
            my_k_mpi_detail::coordinate<StateInteger> const coordinate
              = my_k_mpi_policy::rank_to_coordinate(rank);
#   endif


            // (2) l      <= b < l+ m   (operation for ncqubits):
            // (3) l+ m   <= b < l+ m+3 (operation for nqubits):
            // (4) l+ m+3 <= b < l+2m+3 (operation for ucqubits):
            // (5) l+2m+3 <= b < l+2m+6 (operation for uqubits):
            // (6) l+2m+6 <= b          (operation for gqubits):
#   ifndef KET_USE_CPP03
            constexpr auto condition_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_condition_qubits);
            constexpr auto node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_node_unit_qubits);
#   else
            StateInteger const condition_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_condition_qubits);
            StateInteger const node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_node_unit_qubits);
#   endif

#   ifndef KET_USE_CPP03
            auto const num_xxx_qubits
#   else
            BitInteger const num_xxx_qubits
#   endif
              = operation_for_which_qubit == my_which_qubit::ncqubit
                ? num_lqubits
                : operation_for_which_qubit == my_which_qubit::nqubit
                  ? num_lqubits+num_condition_qubits
                  : operation_for_which_qubit == my_which_qubit::ucqubit
                    ? num_lqubits+num_condition_qubits
                      + num_node_unit_qubits
                    : operation_for_which_qubit == my_which_qubit::uqubit
                      ? num_lqubits+2u*num_condition_qubits
                        + num_node_unit_qubits
                      : num_lqubits+2u*num_condition_qubits
                        + 2u*num_node_unit_qubits;

            // (|...|)001000(|xx...xx|xxxxxxxxxx)
#   ifndef KET_USE_CPP03
            auto const mask
              = permutated_global_swap_qubit_mask >> num_xxx_qubits;
#   else
            StateInteger const mask
              = permutated_global_swap_qubit_mask >> num_xxx_qubits;
#   endif

            using KET_begin;
#   ifndef KET_USE_CPP03
            auto const first = begin(local_state);
#   else
            typedef
              typename boost::range_iterator<LocalState>::type
              local_state_iterator;
            local_state_iterator const first = begin(local_state);
#   endif

            // x,y = u or n (xindex, x_condition, etc.)
            // x probably contains swap qubit, but y doesn't
            // More precisely,
            //   if (is_n_nc_operation(operation_for_which_qubit))
            //   { x = n; y = u; }
            //   else
            //   { x = u; y = n; }
            // Therefore, if swap qubit is in gqubits,
            //neither x nor y contains swap qubit

            // x = 4 (if N_1 < k), X_2 (if X_1 >= k and X_2 <= 3),
            //     X_2+1 (if X_1 >= k and X_2 >= 4)
            // xindex = kX_2+X_1 (if X_1 < k), X_1-k (if X_1 >= k)
            // => xindex = kX_2+X_1 (if x = 4), X_1-k (if x != 4)
            //    X_1 = xindex%k if x=4, xindex+k if x!=4,
            //    X_2 = xindex/k if x=4, x if x<=3, x-1 if x>=5
#   ifndef KET_USE_CPP03
            auto const nindex_last
              = my_k_mpi_detail::is_in_center_node(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;
            auto const uindex_last
              = my_k_mpi_detail::is_in_center_unit(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;
            auto const xindex_last
              = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                ? nindex_last : uindex_last;
            auto const yindex_last
              = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                ? uindex_last : nindex_last;
#   else
            StateInteger const nindex_last
              = my_k_mpi_detail::is_in_center_node(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;
            StateInteger const uindex_last
              = my_k_mpi_detail::is_in_center_unit(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;
            StateInteger const xindex_last
              = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                ? nindex_last : uindex_last;
            StateInteger const yindex_last
              = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                ? uindex_last : nindex_last;
#   endif

#   ifndef KET_USE_CPP03
            for (auto const xindex:
                 boost::irange(StateInteger{0u}, xindex_last))
#   else
            for (StateInteger xindex = 0u; xindex < xindex_last; ++xindex)
#   endif
            {
              // X_1
              // xxbxxx(|xxxxxxxxxx) for xcqubit operation
#   ifndef KET_USE_CPP03
              auto const x_condition_value
#   else
              StateInteger const x_condition_value
#   endif
                = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                  ? my_k_mpi_detail::is_in_center_node(coordinate)
                    ? xindex%node_unit_value_last
                    : xindex+threshold
                  : my_k_mpi_detail::is_in_center_unit(coordinate)
                    ? xindex%node_unit_value_last
                    : xindex+threshold;
              // ~X_1
              // xx(~b)xxx(|xxxxxxxxxx) for xcqubit operation
#   ifndef KET_USE_CPP03
              auto const other_x_condition_value
#   else
              StateInteger const other_x_condition_value
#   endif
                = my_k_mpi_detail::is_condition_operation(
                    operation_for_which_qubit)
                  ? x_condition_value xor mask
                  : x_condition_value;

              // X_2
              // xbx(|xxxxxx|xxxxxxxxxx) for xqubit operation
#   ifndef KET_USE_CPP03
              auto const x_value
#   else
              StateInteger const x_value
#   endif
                = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                  ? my_k_mpi_detail::is_in_center_node(coordinate)
                    ? xindex/node_unit_value_last
                    : coordinate.node()
                        < my_k_mpi_policy::center_node()
                      ? static_cast<StateInteger>(coordinate.node())
                      : static_cast<StateInteger>(
                          coordinate.node()-1u)
                  : my_k_mpi_detail::is_in_center_unit(coordinate)
                    ? xindex/node_unit_value_last
                    : coordinate.unit()
                        < my_k_mpi_policy::center_unit()
                      ? static_cast<StateInteger>(coordinate.unit())
                      : static_cast<StateInteger>(
                          coordinate.unit()-1u);
              // ~X_2
              // x(~b)x(|xxxxxx|xxxxxxxxxx) for xqubit operation
#   ifndef KET_USE_CPP03
              auto const other_x_value
#   else
              StateInteger const other_x_value
#   endif
                = my_k_mpi_detail::is_node_unit_operation(
                    operation_for_which_qubit)
                  ? x_value xor mask
                  : x_value;

              // ~xindex = k~X_2+~X_1 (if ~X_1 < k), ~X_1-k (if ~X_1 >= k)
#   ifndef KET_USE_CPP03
              auto const other_xindex
#   else
              StateInteger const other_xindex
#   endif
                = other_x_condition_value < threshold
                  ? threshold*other_x_value+other_x_condition_value
                  : other_x_condition_value-threshold;

              // ~x = 4 (if ~X_1 < k), ~X_2 (if ~X_1 >= k and ~X_2 <= 3),
              //      ~X_2+1 (if ~X_1 >= k and ~X_2 >= 4)
#   ifndef KET_USE_CPP03
              auto const other_x
#   else
              StateInteger const other_x
#   endif
                = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                  ? other_x_condition_value < threshold
                    ? my_k_mpi_policy::center_node()
                    : other_x_value < my_k_mpi_policy::center_node()
                      ? other_x_value
                      : other_x_value+1u
                  : other_x_condition_value < threshold
                    ? my_k_mpi_policy::center_unit()
                    : other_x_value < my_k_mpi_policy::center_unit()
                      ? other_x_value
                      : other_x_value+1u;


              // if (is_n_nc_operation(operation_for_which_qubit))
              // { x = n; y = u; }
              // else
              // { x = u; y = n; }
#   ifndef KET_USE_CPP03
              auto const other_rank
#   else
              yampi::rank const other_rank
#   endif
                = my_k_mpi_detail::is_g_operation(operation_for_which_qubit)
                  ? my_k_mpi_policy::coordinate_to_rank(
                      coordinate.group() xor mask, coordinate.unit(), coordinate.node())
                  : my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                    ? my_k_mpi_policy::coordinate_to_rank(
                        coordinate.group(), coordinate.unit(), other_x)
                    : my_k_mpi_policy::coordinate_to_rank(
                        coordinate.group(), other_x, coordinate.node());


              // (000000|)b000000000
#   ifndef KET_USE_CPP03
              auto const other_first_lindex
#   else
              StateInteger const other_first_lindex
#   endif
                = my_k_mpi_detail::is_g_operation(operation_for_which_qubit)
                  ? (((coordinate.group() bitand mask) << num_xxx_qubits)
                     >> permutated_global_swap_qubit)
                    << (num_lqubits-1)
                  : my_k_mpi_detail::is_condition_operation(operation_for_which_qubit)
                    ? (((x_condition_value bitand mask) << num_xxx_qubits)
                       >> permutated_global_swap_qubit)
                      << (num_lqubits-1)
                    : (((x_value bitand mask) << num_xxx_qubits)
                       >> permutated_global_swap_qubit)
                      << (num_lqubits-1);
#   ifndef KET_USE_CPP03
              // (000000|)(~b)000000000
              auto const first_lindex
                = other_first_lindex xor permutated_local_swap_qubit_mask;
              // (000000|)(~b)111111111
              auto const prev_last_lindex
                = ((StateInteger{1u} << (num_lqubits-1))
                   - StateInteger{1u})
                  bitor first_lindex;
#   else
              // (000000|)(~b)000000000
              StateInteger const first_lindex
                = other_first_lindex xor permutated_local_swap_qubit_mask;
              // (000000|)(~b)111111111
              StateInteger const prev_last_lindex
                = ((static_cast<StateInteger>(1u) << (num_lqubits-1))
                   - static_cast<StateInteger>(1u))
                  bitor first_lindex;
#   endif

#   ifndef KET_USE_CPP03
              for (auto const yindex:
                   boost::irange(StateInteger{0u}, yindex_last))
#   else
              for (StateInteger yindex = 0u; yindex < yindex_last; ++yindex)
#   endif
              {
                // Y_1
#   ifndef KET_USE_CPP03
                auto const y_condition_value
#   else
                StateInteger const y_condition_value
#   endif
                  = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                    ? my_k_mpi_detail::is_in_center_unit(coordinate)
                      ? yindex%node_unit_value_last
                      : yindex+threshold
                    : my_k_mpi_detail::is_in_center_node(coordinate)
                      ? yindex%node_unit_value_last
                      : yindex+threshold;

                // if (is_n_nc_operation(operation_for_which_qubit))
                // { x = n; y = u; }
                // else
                // { x = u; y = n; }
#   ifndef KET_USE_CPP03
                auto const first_index
#   else
                StateInteger const first_index
#   endif
                  = my_k_mpi_detail::is_n_nc_operation(
                      operation_for_which_qubit)
                    ? my_k_mpi_policy::xindices_to_index(
                        yindex, xindex, first_lindex, x_condition_value)
                    : my_k_mpi_policy::xindices_to_index(
                        xindex, yindex, first_lindex, y_condition_value);
#   ifndef KET_USE_CPP03
                auto const last_index
#   else
                StateInteger const last_index
#   endif
                  = my_k_mpi_detail::is_n_nc_operation(
                      operation_for_which_qubit)
                    ? my_k_mpi_policy::xindices_to_index(
                        yindex, xindex, prev_last_lindex, x_condition_value)+1
                    : my_k_mpi_policy::xindices_to_index(
                        xindex, yindex, prev_last_lindex, y_condition_value)+1;

                if (rank == other_rank)
                {
#   ifndef KET_USE_CPP03
                  auto const other_first_index
#   else
                  StateInteger const other_first_index
#   endif
                    = my_k_mpi_detail::is_n_nc_operation(operation_for_which_qubit)
                      ? my_k_mpi_policy::xindices_to_index(
                          yindex, other_xindex, other_first_lindex,
                          other_x_condition_value)
                      : my_k_mpi_policy::xindices_to_index(
                          other_xindex, yindex, other_first_lindex,
                          y_condition_value);

                  std::swap_ranges(
                    first+first_index, first+last_index,
                    first+other_first_index);
                }
                else
                {
                  using KET_begin;
                  yampi::send_receive(
                    begin(local_state)+first_index, last_index-first_index,
                    rank, yampi::tag(0), other_rank, yampi::tag(0),
                    communicator, yampi::ignore_status);
                }
              }
            }


            using ket::mpi::permutate;
            permutate(permutation, qubits[0u], local_swap_qubit);
          }
        };


#   ifndef KET_USE_CPP03
        template <typename MpiPolicy>
        struct for_each_local_range
        {
          template <typename LocalState, typename Function>
          static LocalState& call(LocalState& local_state, Function&& function);
        };

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger = KET_uint64_t>
        struct for_each_local_range<
          ket::mpi::utility::policy::k_mpi<
            num_condition_qubits, threshold, StateInteger>>
        {
          template <typename LocalState, typename Function>
          static LocalState& call(LocalState& local_state, Function&& function)
          {
            using std::begin;
            auto const local_state_first = begin(local_state);

            // coordinate := (group, unit, node)
            using my_k_mpi_policy
              = ket::mpi::utility::policy::k_mpi<
                  num_condition_qubits, threshold, StateInteger>;
            auto const coordinate = my_k_mpi_policy::coordinate();

            constexpr auto num_node_unit_qubits
              = my_k_mpi_policy::num_node_unit_qubits();
            constexpr auto node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_node_unit_qubits);
            auto constexpr condition_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_condition_qubits);

            // node = 4 (if N_1 < k), N_2 (if N_1 >= k and N_2 <= 3),
            //        N_2+1 (if N_1 >= k and N_2 >= 4)
            // nindex = kN_2+N_1 (if N_1 < k), N_1-k (if N_1 >= k)
            // => nindex = kN_2+N_1 (if node = 4), N_1-k (if node != 4)
            namespace my_k_mpi_detail = ket::mpi::utility::k_mpi_detail;
            auto const nindex_last
              = my_k_mpi_detail::is_in_center_node(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;

            // unit = 4 (if U_1 < k), U_2 (if U_1 >= k and U_2 <= 3),
            //        U_2+1 (if U_1 >= k and U_2 >= 4)
            // uindex = kU_2+U_1 (if U_1 < k), U_1-k (if U_1 >= k)
            // => uindex = kU_2+U_1 (if unit = 4), U_1-k (if unit != 4)
            auto const uindex_last
              = my_k_mpi_detail::is_in_center_unit(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;

            for (auto const index:
                 boost::irange(StateInteger{0u}, nindex_last*uindex_last))
            {
              auto const last_lindex
                = ket::utility::integer_exp2<StateInteger>(
                    my_k_mpi_policy::num_lqubits());

              function(
                local_state_first+index*last_lindex,
                local_state_first+(index+StateInteger{1u})*last_lindex);
            }

            return local_state;
          }
        };
#   else
        template <typename MpiPolicy>
        struct for_each_local_range;

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        struct for_each_local_range<
          ket::mpi::utility::policy::k_mpi<
            num_condition_qubits, threshold, StateInteger> >
        {
          template <typename LocalState, typename Function>
          static LocalState& call(LocalState& local_state, Function function)
          {
            //
            {
            typedef
              ket::mpi::utility::policy::k_mpi<
                num_condition_qubits, threshold, StateInteger>
              my_k_mpi_policy;
            if (my_k_mpi_policy::mpi_rank() == 0u)
            {
              std::cout << "size(local_state)=" << boost::size(local_state) << std::endl;
              std::cout << "for_each_local_range start" << std::endl;
            }
            }
            //
            using boost::begin;
            typedef
              typename boost::range_iterator<LocalState>::type
              local_state_iterator;
            local_state_iterator const local_state_first = begin(local_state);

            // coordinate := (group, unit, node)
            typedef
              ket::mpi::utility::policy::k_mpi<
                num_condition_qubits, threshold, StateInteger>
              my_k_mpi_policy;
            typedef
              ket::mpi::utility::k_mpi_detail::coordinate<StateInteger>
              coordinate_type;
            coordinate_type const coordinate = my_k_mpi_policy::coordinate();

            std::size_t const num_node_unit_qubits
              = my_k_mpi_policy::num_node_unit_qubits();
            StateInteger const node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_node_unit_qubits);
            StateInteger const condition_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_condition_qubits);

            // node = 4 (if N_1 < k), N_2 (if N_1 >= k and N_2 <= 3),
            //        N_2+1 (if N_1 >= k and N_2 >= 4)
            // nindex = kN_2+N_1 (if N_1 < k), N_1-k (if N_1 >= k)
            // => nindex = kN_2+N_1 (if node = 4), N_1-k (if node != 4)
            namespace my_k_mpi_detail = ket::mpi::utility::k_mpi_detail;
            StateInteger const nindex_last
              = my_k_mpi_detail::is_in_center_node(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;

            // unit = 4 (if U_1 < k), U_2 (if U_1 >= k and U_2 <= 3),
            //        U_2+1 (if U_1 >= k and U_2 >= 4)
            // uindex = kU_2+U_1 (if U_1 < k), U_1-k (if U_1 >= k)
            // => uindex = kU_2+U_1 (if unit = 4), U_1-k (if unit != 4)
            StateInteger const uindex_last
              = my_k_mpi_detail::is_in_center_unit(coordinate)
                ? static_cast<StateInteger>(
                    node_unit_value_last*threshold)
                : condition_value_last-threshold;

            //
            if (my_k_mpi_policy::mpi_rank() == 0u)
            {
              std::cout << "nindex_last=" << nindex_last << ", uindex_last=" << uindex_last << std::endl;
              std::cout << "for_each_local_range loop start" << std::endl;
            }
            //
            for (StateInteger index = 0u; index < nindex_last*uindex_last; ++index)
            {
              StateInteger const last_lindex
                = ket::utility::integer_exp2<StateInteger>(
                    my_k_mpi_policy::num_lqubits());
            //
            if (my_k_mpi_policy::mpi_rank() == 0u)
              std::cout << "index=" << index << ", index*last_lindex=" << index*last_lindex << ", (index+1u)*last_lindex=" << (index+1u)*last_lindex << std::endl;
            //

              function(
                local_state_first+index*last_lindex,
                local_state_first+(index+static_cast<StateInteger>(1u))*last_lindex);
            }

            return local_state;
          }
        };
#   endif


        template <typename MpiPolicy>
        struct rank_index_to_qubit_value;

        template <
          std::size_t num_condition_qubits, std::size_t threshold,
          typename StateInteger>
        struct rank_index_to_qubit_value<
          ket::mpi::utility::policy::k_mpi<
            num_condition_qubits, threshold, StateInteger> >
        {
          template <typename LocalState>
          static StateInteger call(
            LocalState const&,
            yampi::rank const rank, StateInteger const index)
          {
            namespace my_k_mpi_detail = ket::mpi::utility::k_mpi_detail;
#   ifndef KET_USE_CPP03
            using my_k_mpi_policy = ket::mpi::utility::policy::k_mpi<
              num_condition_qubits, threshold, StateInteger>;
            auto const coordinate = my_k_mpi_policy::rank_to_coordinate(rank);
            auto const xindices = my_k_mpi_policy::index_to_xindices(index, coordinate);
#   else
            typedef
              ket::mpi::utility::policy::k_mpi<
                num_condition_qubits, threshold, StateInteger>
              my_k_mpi_policy;
            typedef
              my_k_mpi_detail::coordinate<StateInteger>
              coordinate_type;
            coordinate_type const coordinate = my_k_mpi_policy::rank_to_coordinate(rank);
            typedef
              my_k_mpi_detail::xindices<StateInteger>
              xindices_type;
            xindices_type const xindices = my_k_mpi_policy::index_to_xindices(index, coordinate);
#   endif

#   ifndef KET_USE_CPP03
            constexpr auto num_node_unit_qubits
              = my_k_mpi_policy::num_node_unit_qubits();
            constexpr auto node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_node_unit_qubits);
#   else
            std::size_t const num_node_unit_qubits
              = my_k_mpi_policy::num_node_unit_qubits();
            StateInteger const node_unit_value_last
              = ket::utility::integer_exp2<StateInteger>(
                  num_node_unit_qubits);
#   endif

            // N_1
#   ifndef KET_USE_CPP03
            auto const node_condition_value
#   else
            StateInteger const node_condition_value
#   endif
              = my_k_mpi_detail::is_in_center_node(coordinate)
                ? xindices.nindex()%node_unit_value_last
                : xindices.nindex()+threshold;
            // U_1
#   ifndef KET_USE_CPP03
            auto const unit_condition_value
#   else
            StateInteger const unit_condition_value
#   endif
              = my_k_mpi_detail::is_in_center_unit(coordinate)
                ? xindices.uindex()%node_unit_value_last
                : xindices.uindex()+threshold;

            // N_2
#   ifndef KET_USE_CPP03
            auto const node_value
#   else
            StateInteger const node_value
#   endif
              = my_k_mpi_detail::is_in_center_node(coordinate)
                ? xindices.nindex()/node_unit_value_last
                : coordinate.node() < my_k_mpi_policy::center_node()
                  ? static_cast<StateInteger>(coordinate.node())
                  : static_cast<StateInteger>(
                      coordinate.node()-1u);
            // U_2
#   ifndef KET_USE_CPP03
            auto const unit_value
#   else
            StateInteger const unit_value
#   endif
              = my_k_mpi_detail::is_in_center_unit(coordinate)
                ? xindices.uindex()/node_unit_value_last
                : coordinate.unit() < my_k_mpi_policy::center_unit()
                  ? static_cast<StateInteger>(coordinate.unit())
                  : static_cast<StateInteger>(
                      coordinate.unit()-1u);

#   ifndef KET_USE_CPP03
            auto const num_lqubits = my_k_mpi_policy::num_lqubits();
#   else
            std::size_t const num_lqubits = my_k_mpi_policy::num_lqubits();
#   endif

            return 
              xindices.lindex()
              + node_condition_value << num_lqubits
              + node_value << (num_lqubits+num_condition_qubits)
              + unit_condition_value << (num_lqubits+num_condition_qubits+num_node_unit_qubits)
              + unit_value << (num_lqubits+2u*num_condition_qubits+num_node_unit_qubits)
              + coordinate.group() << (num_lqubits+2u*num_condition_qubits+2u*num_node_unit_qubits);
          }
        };


        // TODO: implement specialization of qubit_value_to_rank_index
        template <typename MpiPolicy>
        struct qubit_value_to_rank_index;
      }
    }
  }
}


#   undef KET_uint64_t
#   undef KET_is_unsigned
#   undef KET_array
#   undef KET_begin
#   undef KET_end
#   ifdef KET_USE_CPP03
#     undef static_assert
#     undef noexcept
#     undef constexpr
#   endif
# else // __FUJITSU
#   include <ket/mpi/utility/general_mpi.hpp>

#   ifdef KET_USE_CPP03
#     define noexcept throw()
#     define constexpr 
#   endif


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      namespace policy
      {
#   ifndef KET_USE_CPP03
        using k_mpi = ket::mpi::utility::policy::general_mpi;
#   else
        typedef ket::mpi::utility::policy::general_mpi k_mpi;
#   endif

        inline constexpr k_mpi make_k_mpi() noexcept
        {
#   ifndef KET_USE_CPP03
          return k_mpi{};
#   else
          return k_mpi();
#   endif
        }
      }
    }
  }
}


#   ifdef KET_USE_CPP03
#     undef noexcept
#     undef constexpr
#   endif
# endif // __FUJITSU

#endif

