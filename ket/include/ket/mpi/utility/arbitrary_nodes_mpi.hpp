#ifndef KET_MPI_UTILITY_ARBITRARY_NODES_MPI
# define KET_MPI_UTILITY_ARBITRARY_NODES_MPI

# include <boost/config.hpp>

# include <cstddef>
# include <cassert>
# include <stdexcept>

# include <yampi/communicator.hpp>
# include <yampi/rank.hpp>

# include <ket/utility/integer_exp2.hpp>
# include <ket/utility/integer_log2.hpp>


namespace ket
{
  namespace mpi
  {
    namespace utility
    {
      class invalid_node_units_mpi_initialization_error
        : public std::logic_error
      {
#   ifndef BOOST_NO_CXX11_TEMPLATE_ALIASES
        using super_type = std::logic_error;
#   else
        typedef std::logic_error super_type;
#   endif

       public:
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
        invalid_node_units_mpi_initialization_error(char const* argument)
          : super_type{argument}
        { }

        invalid_node_units_mpi_initialization_error(std::string const& argument)
          : super_type{argument}
        { }
#   else
        invalid_node_units_mpi_initialization_error(char const* argument)
          : super_type(argument)
        { }

        invalid_node_units_mpi_initialization_error(std::string const& argument)
          : super_type(argument)
        { }
#   endif
      };

      namespace policy
      {
        template <std::size_t num_unit_qubits_, std::size_t unit_size_>
        class node_units_mpi
        {
          static bool is_initialized_;

          static int mpi_size_;
          static yampi::rank mpi_rank_;

          static std::size_t num_units_;

          static std::size_t num_qubits_;
          static std::size_t num_gqubits_;
          static std::size_t num_lqubits_;

          static std::size_t local_state_size_;

         public:
          node_units_mpi(std::size_t const num_qubits, yampi::communicator const communicator)
          {
            if (is_initialized_)
              return;

            mpi_size_ = communicator.size();
            mpi_rank_ = communicator.rank();

            num_units_ = mpi_size_ / unit_size_;
            if (mpi_size_ % unit_size_ != 0
                or ::ket::utility::integer_exp2<std::size_t>(
                     ::ket::utility::integer_log2<std::size_t>(num_units_))
                   != num_units_)
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
              throw ::ket::mpi::utility::invalid_node_units_mpi_initialization_error{
                "wrong number of MPI processes created"};
#   else
              throw ::ket::mpi::utility::invalid_node_units_mpi_initialization_error(
                "wrong number of MPI processes created");
#   endif

            num_qubits_ = num_qubits;
            num_gqubits_ = ::ket::utility::integer_log2<std::size_t>(num_units_);

            if (num_qubits_ <= num_gqubits_+num_unit_qubits_)
#   ifndef BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
              throw ::ket::mpi::utility::invalid_node_units_mpi_initialization_error{
                "wrong number of MPI processes created"};
#   else
              throw ::ket::mpi::utility::invalid_node_units_mpi_initialization_error(
                "wrong number of MPI processes created");
#   endif
            num_lqubits_ = num_qubits_-(num_gqubits_+num_unit_qubits_);

            local_state_size_ = local_state_size(mpi_rank_);

            is_initialized_ = true;
          }

          static int mpi_size() { assert(is_initialized_); return mpi_size_; }
          static yampi::rank mpi_rank() { assert(is_initialized_); return mpi_rank_; }

          static std::size_t num_units() { assert(is_initialized_); return num_units_; }
          static BOOST_CONSTEXPR std::size_t unit_size() { return unit_size_; }

          static std::size_t num_qubits() { assert(is_initialized_); return num_qubits_; }
          static std::size_t num_gqubits() { assert(is_initialized_); return num_gqubits_; }
          static BOOST_CONSTEXPR std::size_t num_unit_qubits() { return num_unit_qubits_; }
          static std::size_t num_lqubits() { assert(is_initialized_); return num_lqubits_; }


         private:
          std::size_t local_state_size(yampi::rank const rank) const
          {
            auto const size
              = ::ket::utility::integer_exp2<std::size_t>(num_unit_qubits_) / unit_size_;
            auto const remainder
              = ::ket::utility::integer_exp2<std::size_t>(num_unit_qubits_) % unit_size_;

            a
          }
        };

        std::size_t node_units_mpi::num_qubits_ = 1u;
        bool node_units_mpi::is_initialized_ = false;
      }
    }
  }
}


#endif

