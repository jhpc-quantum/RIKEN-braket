#ifndef BRA_NO_MPI

# include <memory>
# include <string>

# include <yampi/broadcast.hpp>
# include <yampi/buffer.hpp>
# include <yampi/gather.hpp>
# include <yampi/rank.hpp>
# include <yampi/receive.hpp>
# include <yampi/scatter.hpp>
# include <yampi/send.hpp>
# include <yampi/tag.hpp>

# include <bra/mpi_state.hpp>

namespace bra
{
  auto mpi_state::do_send_real_variable(
    int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index == circuit_index_)
      return;
    if (is_real_symbol(variable_name))
      return;

    auto const& real_variable = to_real_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    auto const tag
      = yampi::tag{circuit_index_ + num_circuits * destination_circuit_index + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::send(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      yampi::rank{destination_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_send_complex_variable(
    int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index == circuit_index_)
      return;
    if (is_complex_symbol(variable_name))
      return;

    auto const& complex_variable = to_complex_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    auto const tag
      = yampi::tag{circuit_index_ + num_circuits * destination_circuit_index + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::send(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      yampi::rank{destination_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_send_int_variable(
    int const destination_circuit_index, std::string const& variable_name, int const num_elements) const -> void
  {
    if (destination_circuit_index == circuit_index_)
      return;
    if (is_int_symbol(variable_name))
      return;

    auto const& int_variable = to_int_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    auto const tag
      = yampi::tag{circuit_index_ + num_circuits * destination_circuit_index + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::send(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      yampi::rank{destination_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_receive_real_variable(
    int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (source_circuit_index == circuit_index_)
      return;
    if (is_real_symbol(variable_name))
      return;

    auto& real_variable = to_real_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    auto const tag
      = yampi::tag{source_circuit_index + num_circuits * circuit_index_ + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::receive(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      yampi::rank{source_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_receive_complex_variable(
    int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (source_circuit_index == circuit_index_)
      return;
    if (is_complex_symbol(variable_name))
      return;

    auto& complex_variable = to_complex_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    auto const tag
      = yampi::tag{source_circuit_index + num_circuits * circuit_index_ + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::receive(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      yampi::rank{source_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_receive_int_variable(
    int const source_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (source_circuit_index == circuit_index_)
      return;
    if (is_int_symbol(variable_name))
      return;

    auto& int_variable = to_int_variable(variable_name);
    auto const num_circuits = intercircuit_communicator_.size(environment_);
    auto const tag
      = yampi::tag{source_circuit_index + num_circuits * circuit_index_ + num_circuits * num_circuits * circuit_communicator_.size(environment_)};

    yampi::receive(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      yampi::rank{source_circuit_index}, tag, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_broadcast_real_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (is_real_symbol(variable_name))
      return;

    auto& real_variable = to_real_variable(variable_name);
    yampi::broadcast(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      yampi::rank{root_circuit_index}, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_broadcast_complex_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (is_complex_symbol(variable_name))
      return;

    auto& complex_variable = to_complex_variable(variable_name);
    yampi::broadcast(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      yampi::rank{root_circuit_index}, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_broadcast_int_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements) -> void
  {
    if (is_int_symbol(variable_name))
      return;

    auto& int_variable = to_int_variable(variable_name);
    yampi::broadcast(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      yampi::rank{root_circuit_index}, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_gather_real_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements,
    std::string const& destination_variable_name) -> void
  {
    if (is_real_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (destination_variable_name == "")
      {
        auto& real_variable = to_real_variable(variable_name);
        yampi::gather(
          yampi::in_place,
          yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);
        return;
      }

      auto const& real_variable = to_real_variable(variable_name);
      auto& destination_real_variable = to_real_variable(destination_variable_name);
      yampi::gather(
        yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
        std::addressof(destination_real_variable),
        intercircuit_root, intercircuit_communicator_, environment_);
      return;
    }

    auto const& real_variable = to_real_variable(variable_name);
    yampi::gather(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_gather_complex_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements,
    std::string const& destination_variable_name) -> void
  {
    if (is_complex_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (destination_variable_name == "")
      {
        auto& complex_variable = to_complex_variable(variable_name);
        yampi::gather(
          yampi::in_place,
          yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);
        return;
      }

      auto const& complex_variable = to_complex_variable(variable_name);
      auto& destination_complex_variable = to_complex_variable(destination_variable_name);
      yampi::gather(
        yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
        std::addressof(destination_complex_variable),
        intercircuit_root, intercircuit_communicator_, environment_);
      return;
    }

    auto const& complex_variable = to_complex_variable(variable_name);
    yampi::gather(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_gather_int_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements,
    std::string const& destination_variable_name) -> void
  {
    if (is_int_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (destination_variable_name == "")
      {
        auto& int_variable = to_int_variable(variable_name);
        yampi::gather(
          yampi::in_place,
          yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);
        return;
      }

      auto const& int_variable = to_int_variable(variable_name);
      auto& destination_int_variable = to_int_variable(destination_variable_name);
      yampi::gather(
        yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
        std::addressof(destination_int_variable),
        intercircuit_root, intercircuit_communicator_, environment_);
      return;
    }

    auto const& int_variable = to_int_variable(variable_name);
    yampi::gather(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_scatter_real_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements,
    std::string const& source_variable_name) -> void
  {
    if (is_real_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (source_variable_name == "")
      {
        auto const& real_variable = to_real_variable(variable_name);
        yampi::scatter(
          yampi::in_place,
          yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);
        return;
      }

      auto& real_variable = to_real_variable(variable_name);
      auto const& source_real_variable = to_real_variable(source_variable_name);
      yampi::scatter(
        std::addressof(source_real_variable),
        yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
        intercircuit_root, intercircuit_communicator_, environment_);
      return;
    }

    auto& real_variable = to_real_variable(variable_name);
    yampi::scatter(
      yampi::make_buffer(std::addressof(real_variable), std::addressof(real_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_scatter_complex_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements,
    std::string const& source_variable_name) -> void
  {
    if (is_complex_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (source_variable_name == "")
      {
        auto const& complex_variable = to_complex_variable(variable_name);
        yampi::scatter(
          yampi::in_place,
          yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);
        return;
      }

      auto& complex_variable = to_complex_variable(variable_name);
      auto const& source_complex_variable = to_complex_variable(source_variable_name);
      yampi::scatter(
        std::addressof(source_complex_variable),
        yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
        intercircuit_root, intercircuit_communicator_, environment_);
      return;
    }

    auto& complex_variable = to_complex_variable(variable_name);
    yampi::scatter(
      yampi::make_buffer(std::addressof(complex_variable), std::addressof(complex_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }

  auto mpi_state::do_scatter_int_variable(
    int const root_circuit_index, std::string const& variable_name, int const num_elements,
    std::string const& source_variable_name) -> void
  {
    if (is_int_symbol(variable_name))
      return;

    auto const intercircuit_root = yampi::rank{root_circuit_index};
    if (intercircuit_communicator_.rank(environment_) == intercircuit_root)
    {
      if (source_variable_name == "")
      {
        auto const& int_variable = to_int_variable(variable_name);
        yampi::scatter(
          yampi::in_place,
          yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements * intercircuit_communicator_.size(environment_)),
          intercircuit_root, intercircuit_communicator_, environment_);
        return;
      }

      auto& int_variable = to_int_variable(variable_name);
      auto const& source_int_variable = to_int_variable(source_variable_name);
      yampi::scatter(
        std::addressof(source_int_variable),
        yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
        intercircuit_root, intercircuit_communicator_, environment_);
      return;
    }

    auto& int_variable = to_int_variable(variable_name);
    yampi::scatter(
      yampi::make_buffer(std::addressof(int_variable), std::addressof(int_variable) + num_elements),
      intercircuit_root, intercircuit_communicator_, environment_);
  }
} // namespace bra

#endif // BRA_NO_MPI
