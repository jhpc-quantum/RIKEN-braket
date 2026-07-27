#ifndef BRA_MPI_STATE_HPP
# define BRA_MPI_STATE_HPP

# ifndef BRA_NO_MPI

#   include <string>
#   include <vector>

#   include <yampi/communicator.hpp>
#   include <yampi/environment.hpp>
#   include <yampi/intercommunicator.hpp>

#   include <bra/state.hpp>

namespace bra
{
  class mpi_state
    : public ::bra::state
  {
   protected:
    using ::bra::state::state;

    ~mpi_state() = default;
    mpi_state(mpi_state const&) = default;
    mpi_state& operator=(mpi_state const&) = default;
    mpi_state(mpi_state&&) = default;
    mpi_state& operator=(mpi_state&&) = default;

   private:
    auto do_send_real_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void override;
    auto do_send_complex_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void override;
    auto do_send_int_variable(int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void override;
    auto do_receive_real_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void override;
    auto do_receive_complex_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void override;
    auto do_receive_int_variable(int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void override;
    auto do_broadcast_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void override;
    auto do_broadcast_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void override;
    auto do_broadcast_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void override;
    auto do_gather_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void override;
    auto do_gather_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void override;
    auto do_gather_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& destination_variable_name) -> void override;
    auto do_scatter_real_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void override;
    auto do_scatter_complex_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void override;
    auto do_scatter_int_variable(int const root_circuit_index, std::string const& variable_name, int const num_elements, std::string const& source_variable_name) -> void override;
  }; // class mpi_state
} // namespace bra

# endif // BRA_NO_MPI

#endif // BRA_MPI_STATE_HPP
